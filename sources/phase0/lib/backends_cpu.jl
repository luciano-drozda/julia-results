# ===== CPU target: used only for tests in the sandbox. The "backend" is STADE's sequential CPU adjoint.
function make_backends()
    be = Dict{String,Backend}()
    be["cpu"] = Backend("cpu", x -> copy(x), () -> nothing, f -> (t = time_ns(); f(); (time_ns() - t) * 1e-9), () -> 0, () -> (GC.gc(); nothing), "")
    return be
end
env_report() = Dict{String,Any}("julia" => string(VERSION), "target" => "cpu")
