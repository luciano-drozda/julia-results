import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_red_rev3_1!(i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(3 - i_n, -1) + 1
        return nothing
    end
    i_x = i_n + (__tid - 1) * -1
    CUDA.@atomic loss[1] += u[i_x] ^ 2
    return nothing
end

function red_rev3_cuda(u, loss, i_n)
    nthread_per_block = 256
    if div(3 - i_n, -1) + 1 < 0
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(3 - i_n, -1) + 1, nthread_per_block)) cuda_kernel_red_rev3_1!(i_n, loss, u)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(u, i_n:-1:3))), abs2, view(u, i_n:-1:3))
            end
    end
    return nothing
end
