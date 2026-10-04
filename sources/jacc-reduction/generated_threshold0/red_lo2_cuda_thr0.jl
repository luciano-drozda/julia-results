import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_red_lo2_1!(i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    CUDA.@atomic loss[1] += u[i_x] ^ 2
    return nothing
end

function red_lo2_cuda(u, loss, i_n)
    nthread_per_block = 256
    if div(i_n - 2, 1) + 1 < 0
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_n - 2, 1) + 1, nthread_per_block)) cuda_kernel_red_lo2_1!(i_n, loss, u)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(u, 2:i_n))), abs2, view(u, 2:i_n))
            end
    end
    return nothing
end
