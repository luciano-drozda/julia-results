# ===== Benchmark job driver (generated payload). All logic lives in functions: no top-level loops.
# Sections: configuration, GPU snapshots, module loading, STADE pass, Python passes, probes, main.

function decode_text(s) ; return String(base64decode(s)); end

const JOB = JSON3.read(decode_text(JOB_B64), Dict{String,Any})
const JOB_ID = JOB["id"]
const WORKROOT = get(ENV, "WORK", tempdir())
const WORK = joinpath(WORKROOT, "bench", JOB_ID)
const REFPATH = joinpath(WORK, "ref.json")
const TOL = Float64(get(JOB, "gate_tol", 1e-9))
const T_START = Ref(time())
over_budget(margin = 0.0) = (b = Float64(get(JOB, "budget_s", 0)); b > 0 && time() - T_START[] > b - margin)

const TORCH_LAUNCH = "source /scratch/coop/drozda/torch-env/bin/activate\npython \"\$@\"\n"
const JAX_LAUNCH = "source /scratch/coop/drozda/jax-env/bin/activate\nSP=\$(python -c \"import site;print(site.getsitepackages()[0])\")\nexport LD_LIBRARY_PATH=\"\$(ls -d \$SP/nvidia/*/lib | tr '\\n' ':')\"\nexport XLA_PYTHON_CLIENT_PREALLOCATE=false\npython \"\$@\"\n"

clean(x) = x
clean(x::AbstractFloat) = isfinite(x) ? x : nothing
clean(x::AbstractDict) = Dict{String,Any}(string(k) => clean(v) for (k, v) in x if k != "samples")
clean(x::AbstractVector) = Any[clean(v) for v in x]
clean(x::Tuple) = Any[clean(v) for v in x]

function now_s(); return time(); end

function plog(msg)
    line = string(round(time() - T_START[]; digits = 1), " s | ", msg)
    println(stderr, line); flush(stderr)
    try; open(joinpath(WORK, "progress.log"), "a") do io; println(io, line); end; catch; end
    return nothing
end

# MPI.Init() hangs in a plain Julia process on this cluster (measured). Jobs that need MPI relaunch themselves under `mpirun -n 1`.
function relaunch_under_mpi()
    launcher = split(get(ENV, "BENCH_LAUNCHER", "mpirun -n 1"))
    script = abspath(PROGRAM_FILE)
    mkpath(WORK); plog("parent: relaunching under " * join(launcher, " "))
    cmd = `$launcher $(Base.julia_cmd()) $script`
    env = merge(Dict(ENV), Dict("BENCH_CHILD" => "1", "BENCH_T0" => string(T_START[])))
    p = run(setenv(ignorestatus(cmd), env))
    plog("parent: child exited with code " * string(p.exitcode))
    exit(p.exitcode)
end

# ------------------------------------------------------------------ GPU snapshots
function smi(query)
    try
        return strip(read(`nvidia-smi $query --format=csv,noheader,nounits`, String))
    catch e
        return "unavailable"
    end
end

function gpu_snapshot(tag)
    d = Dict{String,Any}("tag" => tag, "t" => now_s())
    d["gpu"] = smi("--query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu")
    r = smi("--query-gpu=clocks_throttle_reasons.active")
    r == "unavailable" && (r = smi("--query-gpu=clocks_event_reasons.active"))
    d["throttle"] = r
    d["apps"] = smi("--query-compute-apps=pid,used_memory")
    return d
end

# ------------------------------------------------------------------ generated code
const MODS = Dict{String,Any}()
const LOADS = Any[]

function get_module(fname::String)
    haskey(MODS, fname) && return MODS[fname]
    code = decode_text(SRC_B64[fname])
    t0 = time_ns()
    m = Module(Symbol("G_", replace(fname, "." => "_")))
    Base.include_string(m, code, fname)
    load_s = (time_ns() - t0) * 1e-9
    MODS[fname] = (m, code, load_s)
    push!(LOADS, Dict("kind" => "load", "file" => fname, "load_s" => load_s))
    return MODS[fname]
end

function code_file(fam, backend, mode; variant = "")
    sfx = isempty(variant) ? "" : "_" * variant
    if backend == "cpu"
        return mode == "adjoint" ? fam["kernel"] * "_b.jl" : fam["primal_kernel"] * ".jl"
    end
    suffix = backend
    if mode == "adjoint" || fam["primal_kernel"] == fam["kernel"]
        return fam["kernel"] * "_b_" * suffix * sfx * ".jl"
    end
    return fam["primal_kernel"] * "_" * suffix * sfx * ".jl"
end

# ------------------------------------------------------------------ STADE pass
function load_ref()
    isfile(REFPATH) || return Dict{String,Any}()
    return JSON3.read(read(REFPATH, String), Dict{String,Any})
end

function mem_probe(be, mod, code, fam, arrays, scalars, ints, mode)
    be.reclaim(); free0 = be.memfree()
    st = setup!(be, mod, code, fam, arrays, scalars, ints, mode)
    for _ in 1:3; step!(st); end
    be.sync()
    free1 = be.memfree(); parts = mem_parts(st)
    be.reclaim(); free2 = be.memfree()
    st = nothing; be.reclaim(); free3 = be.memfree()
    return Dict("free0" => free0, "free1" => free1, "free2" => free2, "free3" => free3, "M_A" => parts["total"],
                "M_B" => free0 - free1, "M_C" => free0 - free2, "leak" => free0 - free3, "parts" => parts)
end

function time_parts(st::State, be::Backend)
    r = time_protocol(() -> reset!(st), be; warmup = 3, trials = 8, nevent = 0)
    c = time_protocol(() -> call!(st), be; warmup = 3, trials = 8, nevent = 0)
    med(v) = sort(v)[cld(length(v), 2)]
    return Dict("reset_us" => med(r["trial_us"]), "call_us" => med(c["trial_us"]))
end

function stade_task!(recs, t, ref, BE)
    spec = SPECD[]
    case = first(c for c in spec["cases"] if c["id"] == t["case"])
    fam = spec["families"][case["family"]]
    size = first(s for s in case["sizes"] if s["id"] == t["size"])
    modes = get(t, "modes", ["gate"])
    arrays, scalars, ints = make_data(fam, size["params"])
    azero = String.(get(fam, "analytic_zero", String[]))
    for beK in get(t, "backends", ["cuda", "jacc"])
        be = BE[beK]; contender = (beK == "cuda" ? "S-CUDA" : (beK == "jacc" ? "S-JACC" : "S-CPU")) * (isempty(String(get(t, "stade_variant", ""))) ? "" : "[" * String(t["stade_variant"]) * "]")
        for mode in ("primal", "adjoint")
            get(t, mode, true) || continue
            base = Dict{String,Any}("case" => t["case"], "size" => t["size"], "contender" => contender, "mode" => mode, "scope" => "all")
            try
                variant = String(get(t, "stade_variant", "")); file = code_file(fam, beK, mode; variant = variant)
                (mod, code, load_s) = get_module(file)
                st = setup!(be, mod, code, fam, arrays, scalars, ints, Symbol(mode))
                if beK in ("cuda", "jacc")      # every array of the call must live on the GPU
                    allgpu = all(v -> v isa CuArray, values(st.dev)) && all(v -> v isa CuArray, values(st.shadow))
                    allgpu || error("device arrays are not all CuArray: " * join(unique(string.(typeof.(values(st.dev)))), ", "))
                    base["device_array_type"] = string(typeof(first(values(st.dev))))
                end
                t0 = time_ns(); step!(st); be.sync(); first_s = (time_ns() - t0) * 1e-9
                gate_ok = true
                key = t["case"] * "/" * t["size"] * "/" * mode
                g = mode == "primal" ? gate_primal(st) : gate_adjoint(st)
                flat = mode == "primal" ? g : merge(Dict{String,Any}(g["grads"]), g["loss"] === nothing ? Dict{String,Any}() : Dict{String,Any}("loss" => g["loss"]))
                errs = haskey(ref, key) ? compare_flat(flat, ref[key], azero) : nothing
                mx = errs === nothing || isempty(errs) ? nothing : maximum(values(errs))
                gate_ok = mx === nothing || mx <= TOL
                det = mode == "adjoint" ? g["determinism_err"] : nothing
                push!(recs, merge(base, Dict("kind" => "gate", "sigs" => flat, "errs_vs_ref" => errs, "max_err" => mx,
                                              "passed" => mx === nothing ? nothing : gate_ok, "determinism_err" => det, "gate_call_seconds" => (mode == "adjoint" ? g["call_seconds"] : nothing), "first_call_s" => first_s, "load_s" => load_s)))
                if "time" in modes
                    if gate_ok
                        tm = time_protocol(() -> step!(st), be; warmup = 10, trials = 30, tmin = 0.05, max_call_s = Float64(get(JOB, "max_call_s", 30.0)))
                        tm["parts"] = (mode == "adjoint" && !get(tm, "single_sample", false)) ? time_parts(st, be) : nothing
                        push!(recs, merge(base, tm, Dict("kind" => "time", "first_call_s" => first_s)))
                    else
                        push!(recs, merge(base, Dict("kind" => "skipped", "reason" => "failed the correctness gate", "max_err" => mx)))
                    end
                end
                st = nothing
                if "mem" in modes
                    push!(recs, merge(base, Dict("kind" => "mem"), mem_probe(be, mod, code, fam, arrays, scalars, ints, Symbol(mode))))
                end
            catch e
                push!(recs, merge(base, Dict("kind" => "error", "error" => first(replace(sprint(showerror, e), r"\s+" => " "), 600))))
            end
            be.reclaim()
        end
    end
    return nothing
end

function stade_pass(tasks)
    recs = Any[]; ref = load_ref(); BE = make_backends()
    for t in tasks
        if over_budget(max(30.0, Float64(get(t, "needs_s", 0.0))))
            push!(recs, Dict{String,Any}("kind" => "skipped", "case" => t["case"], "size" => t["size"], "reason" => "time budget (task needs " * string(get(t, "needs_s", 0)) * " s)"))
            continue
        end
        plog("S task " * t["case"] * " " * t["size"]); stade_task!(recs, t, ref, BE)
    end
    return recs
end

# ------------------------------------------------------------------ Python passes
function write_pyfiles()
    d = joinpath(WORK, "lib"); mkpath(d)
    for (name, b64) in PY_B64; write(joinpath(d, name), decode_text(b64)); end
    return d
end

function run_python(fw::String, tasks, tag; write_ref = false, extra = Dict{String,Any}())
    tf = joinpath(WORK, "tasks_" * tag * ".json"); of = joinpath(WORK, "out_" * tag * ".json")
    cfg = Dict{String,Any}("tasks" => tasks, "gate_tol" => TOL, "deadline_unix" => (Float64(get(JOB, "budget_s", 0)) > 0 ? T_START[] + Float64(get(JOB, "budget_s", 0)) - 45.0 : nothing), "time" => get(JOB, "time", Dict()), "cudnn_benchmark" => true)
    merge!(cfg, extra)
    write(tf, JSON3.write(cfg))
    sh = joinpath(WORK, "launch_" * fw * ".sh"); write(sh, get(JOB, "launch_" * fw, fw == "torch" ? TORCH_LAUNCH : JAX_LAUNCH))
    args = ["--fw", fw, "--tasks", tf, "--out", of, "--ref", REFPATH]
    write_ref && push!(args, "--write-ref")
    t0 = time()
    tmo = Float64(get(JOB, "budget_s", 0)) > 0 ? max(60, round(Int, T_START[] + Float64(get(JOB, "budget_s", 0)) - time() - 20)) : 3600
    plog("python " * fw * " starts (timeout " * string(tmo) * " s)")
    p = run(pipeline(ignorestatus(`timeout $tmo bash $sh $(joinpath(WORK, "lib", "harness_fw.py")) $args`);
                     stdout = joinpath(WORK, "stdout_" * tag * ".txt"), stderr = joinpath(WORK, "stderr_" * tag * ".txt")))
    res = Dict{String,Any}("exit" => p.exitcode, "seconds" => time() - t0, "timed_out" => p.exitcode == 124)
    if isfile(of)
        out = JSON3.read(read(of, String), Dict{String,Any})
        res["records"] = out["records"]; res["errors"] = out["errors"]; res["env"] = out["env"]
    else
        res["records"] = Any[]; res["errors"] = Any[Dict("error" => "no output file", "stderr_tail" => tail_of(joinpath(WORK, "stderr_" * tag * ".txt")))]
    end
    return res
end

function tail_of(path; n = 1500)
    isfile(path) || return ""
    s = read(path, String); return s[max(1, end - n):end]
end

function fw_tasks(tasks, fw)
    out = Any[]
    for t in tasks
        get(t, "stade_only", false) && continue
        d = Dict{String,Any}("case" => t["case"], "size" => t["size"], "modes" => get(t, (fw == "torch" ? "torch_modes" : "jax_modes"), get(t, "modes", ["gate"])),
                              "primal" => get(t, "primal", true), "adjoint" => get(t, "adjoint", true))
        if fw == "torch"; haskey(t, "torch_variants") && (d["variants"] = t["torch_variants"])
        else; haskey(t, "jax_variants") && (d["variants"] = t["jax_variants"]); end
        haskey(t, "scopes") && (d["scopes"] = t["scopes"])
        haskey(t, "needs_s") && (d["needs_s"] = t["needs_s"])
        push!(out, d)
    end
    return out
end

# ------------------------------------------------------------------ passes in mirror order
function run_group(gi, group, order)
    all = Any[]; passes = Any[]
    for (pi, letter) in enumerate(order)
        tag = "g$(gi)_p$(pi)_$(letter)"
        plog("pass " * tag * " begins")
        if over_budget(60.0)
            push!(passes, Dict{String,Any}("tag" => tag, "tool" => letter, "skipped" => "time budget")); continue
        end
        snap0 = gpu_snapshot("before_" * tag); t0 = time()
        recs = Any[]; info = Dict{String,Any}("tag" => tag, "tool" => letter)
        if letter == "S"
            recs = stade_pass(group["tasks"])
        else
            fw = letter == "P" ? "torch" : "jax"
            wr = !isfile(REFPATH)
            res = run_python(fw, fw_tasks(group["tasks"], fw), tag; write_ref = wr)
            recs = res["records"]; info["exit"] = res["exit"]; info["python_seconds"] = res["seconds"]; info["env"] = res["env"]
            for e in res["errors"]; push!(recs, merge(Dict("kind" => "error"), e)); end
        end
        for r in recs; r["pass"] = pi; r["tool"] = letter; end
        append!(all, recs)
        info["seconds"] = time() - t0; info["snap_before"] = snap0; info["snap_after"] = gpu_snapshot("after_" * tag)
        push!(passes, info); plog("pass " * tag * " done in " * string(round(info["seconds"]; digits = 1)) * " s, " * string(length(recs)) * " records")
    end
    return all, passes
end

function main()
    T_START[] = haskey(ENV, "BENCH_T0") ? parse(Float64, ENV["BENCH_T0"]) : time()
    mkpath(WORK)
    if get(JOB, "relaunch", false) && !haskey(ENV, "BENCH_CHILD"); relaunch_under_mpi(); end
    plog("main start (child=" * string(haskey(ENV, "BENCH_CHILD")) * ")")
    write_pyfiles(); plog("python files written")
    set_spec!(decode_text(SPEC_B64))
    if get(JOB, "use_mpi", false) && !get(JOB, "stub_mpi", false); MPI.Initialized() || MPI.Init(); end
    res = Dict{String,Any}("job" => JOB_ID, "host" => gethostname(), "plan" => get(JOB, "plan", ""), "stade" => get(JOB, "stade", Dict()))
    res["env"] = env_report(); plog("environment report done")
    res["snap_start"] = gpu_snapshot("start"); plog("first GPU snapshot done")
    t0 = time()
    order = String.(get(JOB, "order", ["P", "S", "J", "J", "S", "P"]))
    allrecs = Any[]; passes = Any[]
    if get(JOB, "probes", false); plog("probes begin"); res["probes"] = run_probes(); plog("probes done"); end
    for (gi, group) in enumerate(JOB["groups"])
        r, p = get(JOB, "kind", "bench") == "train" ? train_group(gi, group, order) : run_group(gi, group, order)
        append!(allrecs, r); append!(passes, p)
    end
    res["passes"] = passes
    res["loads"] = LOADS
    res["records"] = allrecs
    res["seconds"] = time() - t0
    res["snap_end"] = gpu_snapshot("end")
    full = JSON3.write(clean(res))
    mkpath(WORK); write(joinpath(WORK, "raw.json"), full)
    println(full)
end
