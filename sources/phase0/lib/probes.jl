# ===== Preflight probes (job J0 only): MPI single-rank epilogue test and framework capability checks.
function mpi_probe()
    r = Dict{String,Any}()
    try
        MPI.Initialized() || MPI.Init()
        comm = MPI.COMM_WORLD
        r["comm_size"] = MPI.Comm_size(comm); r["cuda_aware_env"] = get(ENV, "OMPI_MCA_opal_cuda_support", "unset")
        spec = SPECD[]; fam = spec["families"]["mlp1d"]; params = first(c for c in spec["cases"] if c["id"] == "K5")["sizes"][1]["params"]
        arrays, scalars, ints = make_data(fam, params)
        be = make_backends()["cuda"]
        (mod, code, _) = get_module("mlp1d_b_cuda.jl")
        st = setup!(be, mod, code, fam, arrays, scalars, ints, :adjoint)
        pnames = String[a["name"] for a in fam["args"] if get(a, "param", false) == true]
        step!(st); be.sync()
        direct = Dict(n => Float64.(Array(st.shadow[n])) for n in pnames); loss_direct = Array(st.dev["loss"])[1]
        ep, names = load_epilogue("mlp1d")
        call_ep(f) = Base.invokelatest(getfield(ep, Symbol("mlp1d_b_" * f)), bind_epilogue(st, names[f], comm, 1)...)
        for (nm, sh) in st.shadow; fill!(sh, 0); end
        fill!(st.shadow["loss"], 1); fill!(st.dev["loss"], 0)
        call_ep("check_replicas")
        call_ep("reset_local!"); call!(st)
        call_ep("allreduce_accum!"); call_ep("allreduce_reduced!"); be.sync()
        viaep = Dict(n => Float64.(Array(st.shadow[n])) for n in pnames)
        r["max_diff_vs_direct"] = maximum(maximum(abs.(viaep[n] .- direct[n])) for n in pnames)
        r["loss_diff"] = abs(Array(st.dev["loss"])[1] - loss_direct)
        r["ok"] = r["max_diff_vs_direct"] < 1e-12
    catch e
        r["ok"] = false; r["error"] = first(replace(sprint(showerror, e), r"\s+" => " "), 600)
    end
    return r
end

function py_probe(fw::String)
    sh = joinpath(WORK, "launch_probe_" * fw * ".sh"); write(sh, get(JOB, "launch_" * fw, fw == "torch" ? TORCH_LAUNCH : JAX_LAUNCH))
    of = joinpath(WORK, "probe_" * fw * ".json")
    p = run(pipeline(ignorestatus(`timeout 150 bash $sh $(joinpath(WORK, "lib", "probe_fw.py")) $fw $of`); stdout = joinpath(WORK, "probe_" * fw * "_stdout.txt"), stderr = joinpath(WORK, "probe_" * fw * "_stderr.txt")))
    res = isfile(of) ? JSON3.read(read(of, String), Dict{String,Any}) : Dict{String,Any}("error" => "no probe output", "stderr_tail" => tail_of(joinpath(WORK, "probe_" * fw * "_stderr.txt")))
    res["exit"] = p.exitcode; res["timed_out"] = p.exitcode == 124
    return res
end

function run_probes()
    d = Dict{String,Any}()
    d["mpi"] = mpi_probe()
    d["torch"] = py_probe("torch")
    d["jax"] = py_probe("jax")
    d["env_vars"] = Dict(k => get(ENV, k, nothing) for k in ("JULIA_DEPOT_PATH", "OMPI_MCA_opal_cuda_support", "JACC_BACKEND", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_PARTITION"))
    return d
end
