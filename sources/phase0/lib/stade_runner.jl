# Generic runner for STADE-generated code. Reads the case spec, builds data identical to the Python side,
# runs primal and adjoint, and implements the gate, timing, and memory protocols.
# The device is reached only through a Backend record, so the same code runs on CPU (tests) and on CUDA/JACC.
using JSON3

struct Backend
    name::String
    wrap::Function          # host array -> device array (copy)
    sync::Function
    elapsed::Function       # f -> seconds measured with device events (or wall time on CPU)
    memfree::Function       # () -> free device bytes
    reclaim::Function       # () -> release cached device memory
    pool::Function          # () -> (bytes in use, bytes cached) of the device memory pool
    fname::String           # suffix used in generated function names: "cuda", "jacc", or "" for the CPU adjoint
end

const SPECD = Ref{Any}(nothing)
set_spec!(text::AbstractString) = (SPECD[] = JSON3.read(text, Dict{String,Any}); SPECD[])

# ------------------------------------------------------------------ data (bit-identical to bench_lib.py)
const C_SEED = 0xD1B54A32D192ED03
const C_BASE = 0x632BE59BD9B4E019
const C_STEP = 0x9E3779B97F4A7C15
function hbits(n::Integer, seed::Integer)
    base = UInt64(seed) * C_SEED + C_BASE
    out = Vector{UInt64}(undef, n)
    Threads.@threads for k in 1:n
        z = UInt64(k - 1) * C_STEP + base
        z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
        out[k] = z ⊻ (z >> 31)
    end
    return out
end
unif(n, seed) = (r = hbits(n, seed); [Float64(x >> 11) * 2.0^-53 for x in r])

function evf(expr, env::Dict{String,Any})
    m = Module(:FormulaEnv)
    for (k, v) in env
        Core.eval(m, :($(Symbol(k)) = $v))
    end
    return Core.eval(m, Meta.parse(String(expr)))
end
roundint(x) = Int(round(x))

function fam_env(fam, params)
    env = Dict{String,Any}(string(k) => v for (k, v) in params)
    dv = get(fam, "derived", nothing)
    dv === nothing || for (k, v) in dv
        env[string(k)] = roundint(evf(v, env))
    end
    return env
end

# returns (arrays, scalars, ints); 2-D arrays are column-major like numpy order="F"
function make_data(fam, params)
    env = fam_env(fam, params); base = SPECD[]["seed_base"]
    arrays = Dict{String,Any}(); scalars = Dict{String,Float64}(); ints = Dict{String,Int}()
    for a in vcat(fam["args"], get(fam, "extra", Any[]))
        nm, kind = a["name"], a["kind"]
        if kind == "int"; ints[nm] = roundint(evf(a["value"], env)); continue; end
        if kind == "fscalar"; scalars[nm] = Float64(evf(a["value"], env)); continue; end
        n = roundint(evf(a["len"], env)); seed = get(a, "seed", 0) * 7919 + base
        if kind == "out"
            arr = zeros(n)
        elseif kind == "in"
            scale = Float64(evf(get(a, "scale", "1"), env)); shift = Float64(evf(get(a, "shift", "0"), env))
            r = unif(n, seed); arr = [scale * (2.0 * x - 1.0) + shift for x in r]
        else   # idx
            rng = roundint(evf(a["range"], env)); b = hbits(n, seed)
            arr = Int64[Int64((x >> 11) % UInt64(rng)) + 1 for x in b]
        end
        shp = get(a, "shape", nothing)
        if shp !== nothing
            arr = reshape(arr, Tuple(roundint(evf(s, env)) for s in shp)...)
        end
        arrays[nm] = arr
    end
    return arrays, scalars, ints
end

# ------------------------------------------------------------------ signatures (same as bench_lib.sig)
function sig_positions(n)
    n <= 1024 && return collect(1:n)
    b = hbits(1024, 999)
    return Int[Int(x % UInt64(n)) + 1 for x in b]
end
function sigj(a)
    flat = vec(Float64.(Array(a))); n = length(flat)
    n == 0 && return Dict("n" => 0, "sum" => 0.0, "sumsq" => 0.0, "maxabs" => 0.0, "samples" => Float64[])
    return Dict("n" => n, "sum" => sum(flat), "sumsq" => sum(abs2, flat), "maxabs" => maximum(abs, flat), "samples" => flat[sig_positions(n)])
end
function sig_err(a, b, floor = 0.0)
    a["n"] != b["n"] && return Inf
    b["n"] == 0 && return 0.0
    fl = max(floor, 1e-300); n = b["n"]
    e_sq = abs(a["sumsq"] - b["sumsq"]) / max(b["sumsq"], n * fl * fl)
    e_sum = abs(a["sum"] - b["sum"]) / max(sqrt(n * b["sumsq"]), n * fl)
    sa, sb = Float64.(a["samples"]), Float64.(b["samples"])
    e_smp = isempty(sa) ? 0.0 : maximum(abs.(sa .- sb)) / max(b["maxabs"], fl)
    e_max = abs(a["maxabs"] - b["maxabs"]) / max(b["maxabs"], fl)
    return max(e_sq, e_sum, e_smp, e_max)
end
function noise_floor(ref_flat; rel = 1e-8)
    scales = [v["maxabs"] for (k, v) in ref_flat if k != "loss" && v !== nothing && v["n"] > 0]
    return isempty(scales) ? 0.0 : rel * maximum(scales)
end

function compare_flat(cand, ref, analytic_zero = String[])
    scales = [v["maxabs"] for (k, v) in ref if k != "loss" && !(k in analytic_zero) && v !== nothing && v["n"] > 0]
    scale = isempty(scales) ? 0.0 : maximum(scales); fl = 1e-8 * scale; errs = Dict{String,Float64}()
    for (k, v) in cand
        (haskey(ref, k) && v !== nothing && ref[k] !== nothing) || continue
        errs[k] = k in analytic_zero ? max(v["maxabs"], ref[k]["maxabs"]) / max(scale, 1e-300) : sig_err(v, ref[k], fl)
    end
    return errs
end

# ------------------------------------------------------------------ device state
mutable struct State
    be::Backend
    mode::Symbol                       # :adjoint or :primal
    fam::Dict{String,Any}
    fname::String                      # kernel function name (without backend suffix)
    argspecs::Vector{Any}
    dev::Dict{String,Any}
    shadow::Dict{String,Any}
    pristine::Dict{String,Any}
    scalars::Dict{String,Float64}
    ints::Dict{String,Int}
    stacks::Tuple
    zero_each::Vector{String}
    reload::Vector{String}
    fn::Any                            # the generated function
    seedshadow::Dict{String,Any}
    ret::Any
end

isfloatarr(a) = a["kind"] in ("in", "out")

function arg_specs(fam, mode)
    if mode == :adjoint
        return Vector{Any}(fam["args"])
    end
    pas = get(fam, "primal_arg_specs", nothing)
    if pas !== nothing; return Vector{Any}(pas); end
    pa = get(fam, "primal_args", nothing)
    pa === nothing && return Vector{Any}(fam["args"])
    byname = Dict(a["name"] => a for a in fam["args"])
    return Any[byname[n] for n in pa]
end

initstack_names(code, fn) = (m = match(Regex("function " * fn * "\\(([^)]*)\\)"), code); m === nothing ? String[] : String[strip(s) for s in split(m.captures[1], ",") if !isempty(strip(s))])

function setup!(be::Backend, mod, code::String, fam, arrays, scalars, ints, mode::Symbol)
    argspecs = arg_specs(fam, mode)
    alias = mode == :primal ? get(fam, "primal_alias", Dict{String,Any}()) : Dict{String,Any}()
    kernel = mode == :adjoint ? fam["kernel"] : fam["primal_kernel"]
    dev = Dict{String,Any}(); shadow = Dict{String,Any}(); pristine = Dict{String,Any}()
    zero_each = String.(get(fam, mode == :adjoint ? "zero_each" : "primal_zero_each", String[]))
    reload = String.(get(fam, mode == :adjoint ? "reload" : "primal_reload", String[]))
    for a in argspecs
        nm = a["name"]
        if a["kind"] in ("in", "out", "idx")
            src = haskey(alias, nm) ? arrays[alias[nm]] : arrays[nm]
            dev[nm] = be.wrap(src)
            if mode == :adjoint && a["kind"] != "idx"
                shadow[nm] = be.wrap(zeros(eltype(src), size(src)))
            end
            nm in reload && (pristine[nm] = be.wrap(src))
        end
    end
    seedshadow = Dict{String,Any}()
    if mode == :adjoint && get(fam, "seed_shadow", nothing) !== nothing
        for (tgt, srcname) in fam["seed_shadow"]
            seedshadow[tgt] = be.wrap(arrays[srcname])
        end
    end
    suffix = isempty(be.fname) ? "" : "_" * be.fname
    fname = mode == :adjoint ? kernel * "_b" * suffix : kernel * suffix
    fn = getfield(mod, Symbol(fname))
    stacks = ()
    if mode == :adjoint
        ini_name = "initstacks_" * kernel * "_b" * suffix
        ini = getfield(mod, Symbol(ini_name))
        argn = initstack_names(code, ini_name)
        st = Base.invokelatest(ini, [ints[n] for n in argn]...)
        stacks = st === nothing ? () : (st isa Tuple ? st : (st,))
    end
    return State(be, mode, fam, kernel, argspecs, dev, shadow, pristine, scalars, ints, stacks, zero_each, reload, fn, seedshadow, nothing)
end

function reset!(s::State)
    for nm in s.zero_each; fill!(s.dev[nm], 0); end
    for nm in s.reload; copyto!(s.dev[nm], s.pristine[nm]); end
    if s.mode == :adjoint
        for (nm, sh) in s.shadow; fill!(sh, 0); end
        lossn = s.fam["loss"]
        lossn === nothing || fill!(s.shadow[lossn], 1)
        for (tgt, sd) in s.seedshadow; copyto!(s.shadow[tgt], sd); end
    end
    return nothing
end

function call!(s::State)
    args = Any[]
    for a in s.argspecs
        nm, kind = a["name"], a["kind"]
        if kind == "int"; push!(args, s.ints[nm])
        elseif kind == "fscalar"
            push!(args, s.scalars[nm]); s.mode == :adjoint && push!(args, 0.0)
        elseif kind == "idx"; push!(args, s.dev[nm])
        else
            push!(args, s.dev[nm]); s.mode == :adjoint && push!(args, s.shadow[nm])
        end
    end
    s.ret = Base.invokelatest(s.fn, args..., s.stacks...)
    return nothing
end

step!(s::State) = (reset!(s); call!(s); nothing)

# scalar shadows come back as a tuple in the order of the float scalar arguments
function scalar_grads(s::State)
    names = [a["name"] for a in s.argspecs if a["kind"] == "fscalar"]
    r = s.ret
    out = Dict{String,Float64}()
    if r isa Tuple && length(r) == length(names)
        for (n, v) in zip(names, r); out[n] = Float64(v); end
    end
    return out
end

# ------------------------------------------------------------------ gate
grad_names(fam, scope) = String[a["name"] for a in fam["args"] if get(a, "grad", false) == true && (scope == "all" || get(a, "param", false) == true)]

function gate_primal(s::State)
    step!(s); s.be.sync()
    return Dict(nm => sigj(s.dev[nm]) for nm in s.fam["gate_out_primal"])
end

function gate_adjoint(s::State; repeats = 5)
    names = grad_names(s.fam, "all"); reps = Any[]; r = 0; nrep = repeats
    while r < nrep
        r += 1
        t0 = time_ns(); step!(s); s.be.sync(); dt = (time_ns() - t0) * 1e-9
        if r == 1                                             # slow call: check determinism fewer times (host-driven code can take minutes)
            dt > 30.0 && (nrep = 1)
            dt > 5.0 && dt <= 30.0 && (nrep = min(nrep, 2))
        end
        grads = Dict{String,Any}(); sg = scalar_grads(s)
        for nm in names
            if haskey(s.shadow, nm); grads[nm] = sigj(s.shadow[nm])
            elseif haskey(sg, nm); grads[nm] = sigj([sg[nm]])
            end
        end
        lossn = s.fam["loss"]
        push!(reps, Dict("grads" => grads, "loss" => lossn === nothing ? nothing : sigj(s.dev[lossn]), "seconds" => dt))
    end
    first = reps[1]; det = 0.0; azero = String.(get(s.fam, "analytic_zero", String[]))
    flat_of(rr) = (f = Dict{String,Any}(rr["grads"]); rr["loss"] === nothing || (f["loss"] = rr["loss"]); f)
    for rr in reps[2:end]
        errs = compare_flat(flat_of(rr), flat_of(first), azero)       # same rule as the gate: analytically zero gradients are compared by magnitude
        isempty(errs) || (det = max(det, maximum(values(errs))))
    end
    return Dict("loss" => first["loss"], "grads" => first["grads"], "determinism_err" => length(reps) > 1 ? det : nothing, "repeats" => length(reps), "call_seconds" => first["seconds"])
end

# ------------------------------------------------------------------ timing
function time_protocol(f!::Function, be::Backend; warmup = 10, trials = 30, tmin = 0.05, kmax = 1000, nevent = 30, max_call_s = 30.0)
    f!(); be.sync()                                            # first call
    t0 = time_ns(); f!(); be.sync(); tq = (time_ns() - t0) * 1e-9
    if tq > max_call_s       # one call is too slow for the job budget: keep this single sample and say so
        return Dict("trial_us" => [tq * 1e6], "event_us" => Float64[], "K" => 1, "trials" => 1, "warmup" => 1, "reduced_protocol" => true, "single_sample" => true)
    end
    reduced = false
    if tq > 0.1; warmup = min(warmup, 2); reduced = true; end
    if tq > 0.5; trials = clamp(round(Int, 15 / tq), 5, trials); nevent = min(nevent, 5); reduced = true; end
    for _ in 1:max(0, warmup - 2); f!(); end
    be.sync()
    t1s = Float64[]
    for _ in 1:3
        t0 = time_ns(); f!(); be.sync(); push!(t1s, (time_ns() - t0) * 1e-9)
    end
    t1 = sort(t1s)[2]
    K = clamp(ceil(Int, tmin / max(t1, 1e-7)), 1, kmax)
    GC.gc()
    tr = Float64[]
    for _ in 1:trials
        t0 = time_ns()
        for _ in 1:K; f!(); end
        be.sync()
        push!(tr, (time_ns() - t0) / K / 1e3)
    end
    ev = Float64[]
    for _ in 1:nevent; push!(ev, be.elapsed(f!) * 1e6); end
    return Dict("trial_us" => tr, "event_us" => ev, "K" => K, "trials" => trials, "warmup" => warmup, "reduced_protocol" => reduced)
end

# ------------------------------------------------------------------ memory
function bytes_of(x)
    x isa AbstractArray ? sizeof(x) : 0
end
function mem_parts(s::State)
    parts = Dict("inputs_params" => 0, "scratch" => 0, "shadows" => 0, "tape" => 0, "pristine_copies" => 0)
    kinds = Dict(a["name"] => a for a in s.argspecs)
    for (nm, d) in s.dev
        k = kinds[nm]
        key = (k["kind"] == "out") ? "scratch" : "inputs_params"
        parts[key] += bytes_of(d)
    end
    for (nm, d) in s.shadow; parts["shadows"] += bytes_of(d); end
    for t in s.stacks; parts["tape"] += bytes_of(t); end
    for (nm, d) in s.pristine; parts["pristine_copies"] += bytes_of(d); end
    for (nm, d) in s.seedshadow; parts["pristine_copies"] += bytes_of(d); end
    parts["total"] = sum(values(parts))
    return parts
end
