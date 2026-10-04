# ===== GPU target: CUDA.jl and JACC.jl backends.
function make_backends()
    be = Dict{String,Backend}()
    be["cuda"] = Backend("cuda", x -> CuArray(x), () -> CUDA.synchronize(), f -> CUDA.@elapsed(f()),
                         () -> Int(CUDA.free_memory()), () -> (GC.gc(); CUDA.reclaim(); nothing), "cuda")
    be["jacc"] = Backend("jacc", x -> JACC.array(x), () -> CUDA.synchronize(), f -> CUDA.@elapsed(f()),
                         () -> Int(CUDA.free_memory()), () -> (GC.gc(); CUDA.reclaim(); nothing), "jacc")
    return be
end

function env_report()
    d = Dict{String,Any}()
    try
        deps = Pkg.project().dependencies
        for nm in ("CUDA", "JACC", "Atomix", "MPI", "JSON3")
            d[nm] = haskey(deps, nm) ? string(Pkg.dependencies()[deps[nm]].version) : "NOT a direct dependency"
        end
    catch e
        d["pkg_error"] = sprint(showerror, e)
    end
    d["julia"] = string(VERSION); d["threads"] = Threads.nthreads()
    d["cuda_driver"] = string(CUDA.driver_version()); d["cuda_runtime"] = string(CUDA.runtime_version()); d["device"] = CUDA.name(CUDA.device())
    return d
end
