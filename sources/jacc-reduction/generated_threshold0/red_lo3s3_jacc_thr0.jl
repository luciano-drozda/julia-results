import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_red_lo3s3_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_red_lo3s3_2!(__jacc_i, i_n, loss, u)
    i_x = 3 + (__jacc_i - 1) * 3
    Atomix.@atomic loss[1] += u[i_x] ^ 2
    return nothing
end

function red_lo3s3_jacc(u, loss, i_n)
    if max(0, div((i_n - 3) + 3, 3)) < 0
        if max(0, div((i_n - 3) + 3, 3)) > 0
            JACC.@parallel_for range = max(0, div((i_n - 3) + 3, 3)) jacc_kernel_red_lo3s3_2!(i_n, loss, u)
        end
    else
        __jgen_redval_1 = JACC.@parallel_reduce(range = max(0, div((i_n - 3) + 3, 3)), (((i_x, u)->u[i_x] ^ 2))(u))
        JACC.@parallel_for range = 1 jacc_kernel_red_lo3s3_1!(loss, __jgen_redval_1)
    end
    return nothing
end
