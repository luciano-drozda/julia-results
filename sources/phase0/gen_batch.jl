include("/home/claude/work/stade_044/STADE.jl/src/STADE.jl"); using .STADE
corp = "/home/claude/work/stade_044/STADE.jl/test/val-corpus"; wrap = "/home/claude/work/phase0/wrap"; out = "/home/claude/work/phase0/final/batch"
for (n, p, ps) in (("mlp1d", "$corp/mlp1d.jl", [:x, :y]), ("transformer_loss", "$wrap/transformer_loss.jl", [:x_in, :target]),
                   ("unet_loss", "$wrap/unet_loss.jl", [:x, :target]), ("mpnn_loss", "$wrap/mpnn_loss.jl", [:node_feat, :edge_feat, :target]))
    STADE.stade_batch_file(p, joinpath(out, n * "_batch.jl"); per_sample = ps, mode = :adjoint)
    println("batch epilogue: ", n)
end
