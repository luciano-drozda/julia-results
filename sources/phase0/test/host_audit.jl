# Count, per call, how many host-driven scalar-access blocks (CUDA.@allowscalar) and kernel launches the generated CUDA host code performs.
# The kernels are NOT executed (mock @cuda); the host control flow is. Counts are exact for the control flow; values are meaningless.
include("../lib/stade_runner.jl")
set_spec!(read("../lib/cases_spec.json", String))
mutable struct Cnt; scalar::Int; launch::Int; end
const CNT = Cnt(0, 0)
function load_mock(code)
    code = replace(code, "using CUDA" => "")
    m = Module(:Mock)
    Core.eval(m, :(const CuArray = Array))
    Core.eval(m, :(macro cuda(args...); :(Main.CNT.launch += 1; nothing); end))
    Core.eval(m, :(module CUDA
        allowscalar(::Bool) = nothing
        macro allowscalar(ex); :(Main.CNT.scalar += 1; $(esc(ex))); end
        macro atomic(ex); nothing; end
    end))
    Base.include_string(m, code, "mock")
    return m, code
end
be = Backend("mock", x -> copy(x), () -> nothing, f -> 0.0, () -> 0, () -> nothing, () -> (0, 0), "cuda")
spec = SPECD[]; final = "/home/claude/work/phase0/final"
rows = []
for c in spec["cases"]
    fam = spec["families"][c["family"]]
    for s in c["sizes"]
        s["id"] == "tiny" && continue
        # skip sizes whose host arrays would be huge (the mock needs real host arrays)
        env = fam_env(fam, s["params"]); tot = sum(roundint(evf(a["len"], env)) for a in fam["args"] if haskey(a, "len"))
        tot > 6e7 && continue
        arrays, scalars, ints = make_data(fam, s["params"])
        for mode in ("adjoint", "primal")
            file = mode == "adjoint" || fam["primal_kernel"] == fam["kernel"] ? fam["kernel"] * "_b_cuda.stripped.jl" : fam["primal_kernel"] * "_cuda.stripped.jl"
            m, code = load_mock(read(joinpath(final, file), String))
            st = setup!(be, m, code, fam, arrays, scalars, ints, Symbol(mode))
            CNT.scalar = 0; CNT.launch = 0
            try; call!(st); catch e; println("ERR ", c["id"], " ", s["id"], " ", mode, " ", first(sprint(showerror, e), 80)); continue; end
            push!(rows, (c["id"], s["id"], mode, CNT.launch, CNT.scalar))
        end
    end
end
println(rpad("case", 5), rpad("size", 13), rpad("mode", 9), rpad("kernel launches", 17), "host scalar-access blocks per call")
for r in rows; println(rpad(r[1], 5), rpad(r[2], 13), rpad(r[3], 9), rpad(string(r[4]), 17), r[5]); end
open("host_audit.json", "w") do io; JSON3.write(io, [Dict("case" => r[1], "size" => r[2], "mode" => r[3], "launches" => r[4], "scalar_blocks" => r[5]) for r in rows]); end
