using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_dotprod_b_1!(i_n, loss, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += u[i_x] * v[i_x]
    return nothing
end

function cuda_kernel_dotprod_b_2!(i_n, lossb, u, ub, v, vb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_x = i_n + (__tid - 1) * -1
    __cse_0 = lossb[1]
    ub[i_x] = ub[i_x] + v[i_x] * __cse_0
    vb[i_x] = vb[i_x] + u[i_x] * __cse_0
    return nothing
end

function cuda_kernel_dotprod_1!(i_n, loss, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += u[i_x] * v[i_x]
    return nothing
end

function initstacks_dotprod_b_cuda()
    return nothing
end

function dotprod_b_cuda(loss, lossb, u, ub, v, vb, i_n)
    nthread_per_block = 256
    if div(i_n - 1, 1) + 1 < 9223372036854775807
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_n - 1, 1) + 1, nthread_per_block)) cuda_kernel_dotprod_b_1!(i_n, loss, u, v)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + mapreduce(init = zero(eltype(view(u, 1:i_n))), ((__mr_1, __mr_2)->__mr_1 * __mr_2), +, view(u, 1:i_n), view(v, 1:i_n))
            end
    end
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(1 - i_n, -1) + 1, nthread_per_block)) cuda_kernel_dotprod_b_2!(i_n, lossb, u, ub, v, vb)
    return nothing
end

function dotprod_cuda(loss, u, v, i_n)
    nthread_per_block = 256
    if div(i_n - 1, 1) + 1 < 9223372036854775807
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_n - 1, 1) + 1, nthread_per_block)) cuda_kernel_dotprod_1!(i_n, loss, u, v)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + mapreduce(init = zero(eltype(view(u, 1:i_n))), ((__mr_1, __mr_2)->__mr_1 * __mr_2), +, view(u, 1:i_n), view(v, 1:i_n))
            end
    end
    return nothing
end
