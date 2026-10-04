include("/home/claude/work/stade_044/STADE.jl/src/STADE.jl"); using .STADE
for n in ("transformer_loss", "mpnn_loss", "unet_loss")
    for (kpp, fii) in ((false, true), (true, false))
        t0 = time()
        try
            r = STADE.stade_validate_from_baseline(:adjoint, "wrap/$(n).jl", "yaml/$(n).yaml"; keep_push_pop = kpp, fuse_ii_loops = fii, trials = 10)
            println(rpad(n, 18), " keep_push_pop=", rpad(kpp, 5), " fuse_ii_loops=", rpad(fii, 5), " -> ", r isa NamedTuple || r isa Dict ? string(r) : repr(r), "  (", round(time() - t0, digits = 1), " s)")
        catch e
            println(rpad(n, 18), " keep_push_pop=", kpp, " FAILED: ", first(replace(sprint(showerror, e), r"\s+" => " "), 300))
        end
    end
end
