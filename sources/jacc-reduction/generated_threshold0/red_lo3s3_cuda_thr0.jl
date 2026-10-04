import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_red_lo3s3_1!(i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > max(0, div((i_n - 3) + 3, 3))
        return nothing
    end
    i_x = 3 + (__tid - 1) * 3
    CUDA.@atomic loss[1] += u[i_x] ^ 2
    return nothing
end

function red_lo3s3_cuda(u, loss, i_n)
    nthread_per_block = 256
    if max(0, div((i_n - 3) + 3, 3)) < 0
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(max(0, div((i_n - 3) + 3, 3)), nthread_per_block)) cuda_kernel_red_lo3s3_1!(i_n, loss, u)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(u, 3:3:i_n))), abs2, view(u, 3:3:i_n))
            end
    end
    return nothing
end
