using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_matvec_loss_b_1!(a, i_m, i_n, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m * i_n - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    __icse_0 = idx - 1
    i_i = div(__icse_0, i_n) + 1
    i_j = mod(__icse_0, i_n) + 1
    CUDA.@atomic v[i_i] += a[i_i, i_j] * u[i_j]
    return nothing
end

function cuda_kernel_matvec_loss_b_2!(i_m, loss, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_i2 = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += v[i_i2] ^ 2
    return nothing
end

function cuda_kernel_matvec_loss_b_3!(i_m, lossb, v, vb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_m, -1) + 1
        return nothing
    end
    i_i2 = i_m + (__tid - 1) * -1
    vb[i_i2] = vb[i_i2] + (2 * v[i_i2]) * lossb[1]
    return nothing
end

function cuda_kernel_matvec_loss_b_4!(a, ab, i_m, i_n, u, ub, vb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_m * i_n, -1) + 1
        return nothing
    end
    idx = i_m * i_n + (__tid - 1) * -1
    __icse_1 = idx - 1
    i_i = div(__icse_1, i_n) + 1
    i_j = mod(__icse_1, i_n) + 1
    __cse_2 = vb[i_i]
    CUDA.@atomic ab[i_i, i_j] += u[i_j] * __cse_2
    CUDA.@atomic ub[i_j] += a[i_i, i_j] * __cse_2
    return nothing
end

function cuda_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m * i_n - 1, 1) + 1
        return nothing
    end
    idx = 1 + (__tid - 1)
    i_i = div(idx - 1, i_n) + 1
    i_j = mod(idx - 1, i_n) + 1
    CUDA.@atomic v[i_i] += a[i_i, i_j] * u[i_j]
    return nothing
end

function cuda_kernel_matvec_loss_2!(i_m, loss, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_i2 = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += v[i_i2] ^ 2
    return nothing
end

function initstacks_matvec_loss_b_cuda()
    return nothing
end

function matvec_loss_b_cuda(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_m * i_n - 1, 1) + 1, nthread_per_block)) cuda_kernel_matvec_loss_b_1!(a, i_m, i_n, u, v)
    if div(i_m - 1, 1) + 1 < 32768
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_m - 1, 1) + 1, nthread_per_block)) cuda_kernel_matvec_loss_b_2!(i_m, loss, v)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(v, 1:i_m))), abs2, view(v, 1:i_m))
            end
    end
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(1 - i_m, -1) + 1, nthread_per_block)) cuda_kernel_matvec_loss_b_3!(i_m, lossb, v, vb)
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(1 - i_m * i_n, -1) + 1, nthread_per_block)) cuda_kernel_matvec_loss_b_4!(a, ab, i_m, i_n, u, ub, vb)
    return nothing
end

function matvec_loss_cuda(loss, a, u, v, i_m, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_m * i_n - 1, 1) + 1, nthread_per_block)) cuda_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    if div(i_m - 1, 1) + 1 < 32768
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_m - 1, 1) + 1, nthread_per_block)) cuda_kernel_matvec_loss_2!(i_m, loss, v)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(v, 1:i_m))), abs2, view(v, 1:i_m))
            end
    end
    return nothing
end
