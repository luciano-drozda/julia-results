using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_mlp1d_b_1!(b1, h1, n_h, w1, x)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_h - 1, 1) + 1
        return nothing
    end
    i_j = 1 + (__tid - 1)
    s1 = w1[i_j] * x[1] + b1[i_j]
    h1[i_j] = tanh(s1)
    return nothing
end

function cuda_kernel_mlp1d_b_2!(b2, h1, h2, n_h, w2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_h - 1, 1) + 1
        return nothing
    end
    i_j2 = 1 + (__tid - 1)
    s2 = b2[i_j2]
    for i_k = 1:n_h
        s2 = s2 + w2[(i_j2 - 1) * n_h + i_k] * h1[i_k]
    end
    h2[i_j2] = tanh(s2)
    return nothing
end

function cuda_kernel_mlp1d_b_3!(h2, h2b, n_h, s3b, w3, w3b)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_h, -1) + 1
        return nothing
    end
    i_k2 = n_h + (__tid - 1) * -1
    w3b[i_k2] = w3b[i_k2] + h2[i_k2] * s3b
    h2b[i_k2] = h2b[i_k2] + w3[i_k2] * s3b
    return nothing
end

function cuda_kernel_mlp1d_b_4!(b2, b2b, h1, h1b, h2b, n_h, w2, w2b)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_h - 1, 1) + 1
        return nothing
    end
    i_j2 = 1 + (__tid - 1)
    s2b = 0.0
    s2 = b2[i_j2]
    for i_k = 1:n_h
        s2 = s2 + w2[(i_j2 - 1) * n_h + i_k] * h1[i_k]
    end
    __oldb_0 = h2b[i_j2]
    h2b[i_j2] = 0.0
    s2b = s2b + (1.0 - tanh(s2) ^ 2) * __oldb_0
    for i_k = 1:n_h
        __cse_2 = w2[(i_j2 - 1) * n_h + i_k]
        __cse_3 = h1[i_k]
        s2 = s2 + __cse_2 * __cse_3
        w2b[(i_j2 - 1) * n_h + i_k] = w2b[(i_j2 - 1) * n_h + i_k] + __cse_3 * s2b
        CUDA.@atomic h1b[i_k] += __cse_2 * s2b
    end
    __oldb_0 = s2b
    s2b = 0.0
    b2b[i_j2] = b2b[i_j2] + __oldb_0
    return nothing
end

function cuda_kernel_mlp1d_b_5!(b1, b1b, h1b, n_h, w1, w1b, x, xb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_h - 1, 1) + 1
        return nothing
    end
    i_j = 1 + (__tid - 1)
    s1b = 0.0
    __cse_4 = w1[i_j]
    __cse_5 = x[1]
    s1 = __cse_4 * __cse_5 + b1[i_j]
    __oldb_0 = h1b[i_j]
    h1b[i_j] = 0.0
    s1b = s1b + (1.0 - tanh(s1) ^ 2) * __oldb_0
    __oldb_0 = s1b
    s1b = 0.0
    w1b[i_j] = w1b[i_j] + __cse_5 * __oldb_0
    CUDA.@atomic xb[1] += __cse_4 * __oldb_0
    b1b[i_j] = b1b[i_j] + __oldb_0
    return nothing
end

function cuda_kernel_mlp1d_1!(b1, h1, n_h, w1, x)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_h - 1, 1) + 1
        return nothing
    end
    i_j = 1 + (__tid - 1)
    s1 = w1[i_j] * x[1] + b1[i_j]
    h1[i_j] = tanh(s1)
    return nothing
end

function cuda_kernel_mlp1d_2!(b2, h1, h2, n_h, w2)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_h - 1, 1) + 1
        return nothing
    end
    i_j2 = 1 + (__tid - 1)
    s2 = b2[i_j2]
    for i_k = 1:n_h
        s2 = s2 + w2[(i_j2 - 1) * n_h + i_k] * h1[i_k]
    end
    h2[i_j2] = tanh(s2)
    return nothing
end

function initstacks_mlp1d_b_cuda(n_h)
    return nothing
end

function mlp1d_b_cuda(loss, lossb, x, xb, y, yb, w1, w1b, b1, b1b, w2, w2b, b2, b2b, w3, w3b, b3, b3b, h1, h1b, h2, h2b, o, ob, n_h)
    nthread_per_block = 256
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s1b = 0.0
    s2b = 0.0
    s3b = 0.0
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(n_h - 1, 1) + 1, nthread_per_block)) cuda_kernel_mlp1d_b_1!(b1, h1, n_h, w1, x)
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(n_h - 1, 1) + 1, nthread_per_block)) cuda_kernel_mlp1d_b_2!(b2, h1, h2, n_h, w2)
    CUDA.@allowscalar begin
            s3 = b3[1]
        end
    if div(n_h - 1, 1) + 1 < 32768
        for i_k2 = 1:n_h
            CUDA.@allowscalar begin
                    s3 = s3 + w3[i_k2] * h2[i_k2]
                end
        end
    else
        s3 = s3 + mapreduce(init = zero(eltype(view(h2, 1:n_h))), ((__mr_1, __mr_2)->__mr_2 * __mr_1), +, view(h2, 1:n_h), view(w3, 1:n_h))
    end
    CUDA.@allowscalar begin
            o[1] = s3
            __cse_0 = o[1] - y[1]
            loss[1] = loss[1] + __cse_0 ^ 2
            __cse_1 = (2__cse_0) * lossb[1]
            ob[1] = ob[1] + __cse_1
            yb[1] = yb[1] + -__cse_1
            __oldb_0 = ob[1]
            ob[1] = 0.0
            s3b = s3b + __oldb_0
        end
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(1 - n_h, -1) + 1, nthread_per_block)) cuda_kernel_mlp1d_b_3!(h2, h2b, n_h, s3b, w3, w3b)
    CUDA.@allowscalar begin
            __oldb_0 = s3b
            s3b = 0.0
            b3b[1] = b3b[1] + __oldb_0
        end
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(n_h - 1, 1) + 1, nthread_per_block)) cuda_kernel_mlp1d_b_4!(b2, b2b, h1, h1b, h2b, n_h, w2, w2b)
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(n_h - 1, 1) + 1, nthread_per_block)) cuda_kernel_mlp1d_b_5!(b1, b1b, h1b, n_h, w1, w1b, x, xb)
    return nothing
end

function mlp1d_cuda(loss, x, y, w1, b1, w2, b2, w3, b3, h1, h2, o, n_h)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(n_h - 1, 1) + 1, nthread_per_block)) cuda_kernel_mlp1d_1!(b1, h1, n_h, w1, x)
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(n_h - 1, 1) + 1, nthread_per_block)) cuda_kernel_mlp1d_2!(b2, h1, h2, n_h, w2)
    CUDA.@allowscalar begin
            s3 = b3[1]
        end
    if div(n_h - 1, 1) + 1 < 32768
        for i_k2 = 1:n_h
            CUDA.@allowscalar begin
                    s3 = s3 + w3[i_k2] * h2[i_k2]
                end
        end
    else
        s3 = s3 + mapreduce(init = zero(eltype(view(h2, 1:n_h))), ((__mr_1, __mr_2)->__mr_2 * __mr_1), +, view(h2, 1:n_h), view(w3, 1:n_h))
    end
    CUDA.@allowscalar begin
            o[1] = s3
            loss[1] = loss[1] + (o[1] - y[1]) ^ 2
        end
    return nothing
end
