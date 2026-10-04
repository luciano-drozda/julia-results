include("/home/claude/work/stade_044/STADE.jl/src/STADE.jl"); using .STADE
corp = "/home/claude/work/stade_044/STADE.jl/test/val-corpus"; wrap = "/home/claude/work/phase0/wrap"; out = "/home/claude/work/phase0/final"
src = Dict("dotprod" => "$corp/dotprod.jl", "stencil_loss" => "$corp/stencil_loss.jl", "matvec_loss" => "$corp/matvec_loss.jl", "advection" => "$corp/advection.jl",
           "mlp1d" => "$corp/mlp1d.jl", "transformer_loss" => "$wrap/transformer_loss.jl", "unet_loss" => "$wrap/unet_loss.jl", "mpnn_loss" => "$wrap/mpnn_loss.jl")
prim = Dict("transformer" => "$corp/transformer.jl", "unet" => "$corp/unet.jl", "mpnn" => "$corp/mpnn.jl")
for (n, p) in src
    b = joinpath(out, n * "_b.jl")
    STADE.stade_adjoint_file(p, b; keep_push_pop = false, fuse_ii_loops = true)
    STADE.stade_cuda_file(b, joinpath(out, n * "_b_cuda.jl"))
    STADE.stade_jacc_file(b, joinpath(out, n * "_b_jacc.jl"))
    println("adjoint+cuda+jacc: ", n)
end
for (n, p) in prim
    STADE.stade_cuda_file(p, joinpath(out, n * "_cuda.jl"))
    STADE.stade_jacc_file(p, joinpath(out, n * "_jacc.jl"))
    println("primal cuda+jacc:  ", n)
end
