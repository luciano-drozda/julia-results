# Ground-truth test: STADE sequential CPU primal/adjoint, driven by the SAME runner code as the GPU job,
# compared with signatures written by the JAX harness (reference file).
include("../lib/stade_runner.jl")
set_spec!(read("../lib/cases_spec.json", String))
ref = JSON3.read(read(ARGS[1], String), Dict{String,Any})
sizeidx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1      # 1 = tiny, 2 = first real size
be = Backend("cpu", x -> copy(x), () -> nothing, f -> (t = time_ns(); f(); (time_ns() - t) * 1e-9), () -> 0, () -> nothing, "")
corp = "/home/claude/work/stade_044/STADE.jl/test/val-corpus"; gen = "/home/claude/work/gen_044"
spec = SPECD[]
for c in spec["cases"]
    fam = spec["families"][c["family"]]
    size = c["sizes"][sizeidx]; sizeid = size["id"]
    arrays, scalars, ints = make_data(fam, size["params"])
    codeA = read(joinpath(gen, fam["kernel"] * "_b.jl"), String)
    codeP = read(joinpath(corp, fam["primal_kernel"] * ".jl"), String)
    modA = Module(Symbol("A_", c["id"])); Base.include_string(modA, codeA, "adj")
    modP = Module(Symbol("P_", c["id"])); Base.include_string(modP, codeP, "pri")
    sP = setup!(be, modP, codeP, fam, arrays, scalars, ints, :primal)
    gp = gate_primal(sP)
    kp = c["id"] * "/" * sizeid * "/primal"
    azero = String.(get(fam, "analytic_zero", String[])); ep = compare_flat(gp, ref[kp], azero)
    sA = setup!(be, modA, codeA, fam, arrays, scalars, ints, :adjoint)
    ga = gate_adjoint(sA)
    ka = c["id"] * "/" * sizeid * "/adjoint"
    cand = Dict{String,Any}(ga["grads"]); ga["loss"] === nothing || (cand["loss"] = ga["loss"])
    ea = compare_flat(cand, ref[ka], azero)
    worst = isempty(ea) ? ("-", 0.0) : first(sort(collect(ea); by = x -> -x[2]))
    println(rpad(c["id"] * " " * c["family"], 20), " primal max err = ", rpad(string(round(maximum(values(ep)); sigdigits = 2)), 9),
            " | adjoint max err = ", rpad(string(round(worst[2]; sigdigits = 2)), 9), " (", worst[1], ") over ", length(ea), " arrays | determinism ", round(ga["determinism_err"]; sigdigits = 2))
end
