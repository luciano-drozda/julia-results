# ===== Training with batch size 1 through the generated stade_batch epilogue (plan section 9).
function make_dataset(fam, params, N)
    env = fam_env(fam, params); base = SPECD[]["seed_base"]; ds = Dict{String,Any}(); lens = Dict{String,Int}()
    for a in fam["args"]
        (a["name"] in fam["per_sample"]) || continue
        n = roundint(evf(a["len"], env))
        scale = Float64(evf(get(a, "scale", "1"), env)); shift = Float64(evf(get(a, "shift", "0"), env))
        r = unif(N * n, a["seed"] * 7919 + base + 1000003)
        ds[a["name"]] = [scale * (2.0 * x - 1.0) + shift for x in r]; lens[a["name"]] = n
    end
    return ds, lens
end

function bind_epilogue(st::State, argnames, comm, nlocal)
    out = Any[]
    for n in argnames
        if n == "comm"; push!(out, comm)
        elseif n == "n_local"; push!(out, nlocal)
        elseif haskey(st.dev, n); push!(out, st.dev[n])
        elseif endswith(n, "b") && haskey(st.shadow, n[1:end-1]); push!(out, st.shadow[n[1:end-1]])
        else; error("cannot bind epilogue argument " * n)
        end
    end
    return out
end

function load_epilogue(kernel::String)
    fname = kernel * "_batch.jl"
    code = decode_text(SRC_B64[fname])
    m = Module(Symbol("EP_", kernel))
    if get(JOB, "stub_mpi", false)      # CPU tests only: single-rank stand-in for MPI.jl
        code = replace(code, "using MPI" => "")
        Core.eval(m, :(module MPI; Allreduce!(x, op, comm) = x; Allreduce(x, op, comm) = x; end))
    end
    Base.include_string(m, code, fname)
    names = Dict{String,Vector{String}}()
    for f in ("reset_accum!", "reset_reduced!", "reset_local!", "allreduce_accum!", "allreduce_reduced!", "check_replicas")
        names[f] = initstack_names(code, kernel * "_b_" * f)
    end
    return m, names
end

function train_stade!(recs, BE, t, comm)
    spec = SPECD[]; TR = spec["train"]
    model = first(m for m in TR["models"] if m["id"] == t["model"]); famname = model["family"]; fam = spec["families"][famname]
    size = first(s for s in model["sizes"] if s["id"] == t["size"])
    N, nwarm, nsteps = TR["N"], TR["nwarm"], TR["nsteps"]; total = nwarm + nsteps; lr = Float64(get(TR["lr"], t["size"], TR["lr"][famname]))
    for beK in get(t, "backends", ["cuda", "jacc"])
        be = BE[beK]; contender = beK == "cuda" ? "S-CUDA" : "S-JACC"
        base = Dict{String,Any}("model" => t["model"], "size" => t["size"], "contender" => contender, "mode" => "train", "lr" => lr)
        try
            arrays, scalars, ints = make_data(fam, size["params"])
            (mod, code, load_s) = get_module(code_file(fam, beK, "adjoint"))
            st = setup!(be, mod, code, fam, arrays, scalars, ints, :adjoint)
            ep, names = load_epilogue(fam["kernel"])
            ds, lens = make_dataset(fam, size["params"], N); ps = String.(fam["per_sample"])
            dsd = Dict(nm => be.wrap(v) for (nm, v) in ds)
            pnames = String[a["name"] for a in fam["args"] if get(a, "param", false) == true]
            init = Dict(n => be.wrap(arrays[n]) for n in pnames)
            kern = fam["kernel"]
            fns = Dict(f => getfield(ep, Symbol(kern * "_b_" * f)) for f in keys(names))
            lossbuf = be.wrap(zeros(total))
            call_ep(f) = Base.invokelatest(fns[f], bind_epilogue(st, names[f], comm, 1)...)
            function reset_all!()
                for n in pnames; copyto!(st.dev[n], init[n]); end
                fill!(lossbuf, 0)
                for (nm, sh) in st.shadow; fill!(sh, 0); end
                fill!(st.shadow["loss"], 1); fill!(st.dev["loss"], 0)
                return nothing
            end
            function load_sample!(k)
                s = (k - 1) % N
                for nm in ps; copyto!(st.dev[nm], 1, dsd[nm], s * lens[nm] + 1, lens[nm]); end
                return nothing
            end
            function one_step!(k)
                load_sample!(k)
                call_ep("reset_local!")
                call!(st)
                call_ep("allreduce_accum!"); call_ep("allreduce_reduced!")
                copyto!(lossbuf, k, st.dev["loss"], 1, 1)
                for nm in pnames; p = st.dev[nm]; g = st.shadow[nm]; p .-= lr .* g; end
                call_ep("reset_accum!"); call_ep("reset_reduced!")
                return nothing
            end
            reset_all!(); load_sample!(1); call_ep("check_replicas")
            stimes = Float64[]; first_s = 0.0
            for k in 1:total
                t0 = time_ns(); one_step!(k); be.sync(); dt = (time_ns() - t0) / 1e3
                k == 1 && (first_s = dt / 1e6)
                k > nwarm && push!(stimes, dt)
            end
            losses = Float64.(Array(lossbuf))
            reset_all!()
            for k in 1:nwarm; one_step!(k); end
            be.sync(); t0 = time_ns()
            for k in (nwarm + 1):total; one_step!(k); end
            be.sync(); tot = (time_ns() - t0) / 1e9
            la = Dict(string(k) => losses[k] for k in (1, 10, 100, 300, total) if k <= total)
            push!(recs, merge(base, Dict("kind" => "train", "first_step_s" => first_s, "sync_step_us" => stimes, "pipelined_total_s" => tot,
                                          "throughput_steps_per_s" => nsteps / tot, "losses_at" => la, "finite" => all(isfinite, losses),
                                          "nsteps" => nsteps, "nwarm" => nwarm, "load_s" => load_s, "mem_parts" => mem_parts(st))))
            st = nothing
        catch e
            push!(recs, merge(base, Dict("kind" => "error", "error" => first(replace(sprint(showerror, e), r"\s+" => " "), 600))))
        end
        be.reclaim()
    end
    return nothing
end

function train_group(gi, group, order)
    all = Any[]; passes = Any[]; BE = make_backends()
    comm = get(JOB, "stub_mpi", false) ? nothing : MPI.COMM_WORLD
    for (pi, letter) in enumerate(order)
        tag = "t$(gi)_p$(pi)_$(letter)"
        cot = wait_for_idle_gpu(90.0); cot["foreign_processes_after_wait"] > 0 && plog("WARNING: other GPU process(es) during pass " * tag)
        snap0 = gpu_snapshot("before_" * tag); t0 = time(); recs = Any[]; info = Dict{String,Any}("tag" => tag, "tool" => letter, "cotenant" => cot)
        if letter == "S"
            for t in group["tasks"]; train_stade!(recs, BE, t, comm); end
        else
            fw = letter == "P" ? "torch" : "jax"
            tasks = Any[]
            for t in group["tasks"]
                vs = get(t, fw * "_variants", Any[nothing])
                for v in vs
                    push!(tasks, v === nothing ? Dict{String,Any}("model" => t["model"], "size" => t["size"]) : Dict{String,Any}("model" => t["model"], "size" => t["size"], "variant" => v))
                end
            end
            res = run_python(fw, tasks, tag)
            recs = res["records"]; info["exit"] = res["exit"]; info["python_seconds"] = res["seconds"]
            for e in res["errors"]; push!(recs, merge(Dict("kind" => "error"), e)); end
        end
        for r in recs; r["pass"] = pi; r["tool"] = letter; end
        append!(all, recs)
        info["seconds"] = time() - t0; info["snap_before"] = snap0; info["snap_after"] = gpu_snapshot("after_" * tag)
        push!(passes, info)
    end
    return all, passes
end
