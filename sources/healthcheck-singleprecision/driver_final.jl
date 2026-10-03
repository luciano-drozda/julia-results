# Single-precision health check. Runs STADE-generated CUDA and JACC adjoints in Float64 and Float32,
# then calls the PyTorch and JAX harnesses on the same inputs. Prints one JSON line at the end.
using CUDA, JSON3
import JACC

const SRC = Dict{String,Dict{String,String}}()   # filled below by the payload builder
SRC["stencil_loss"] = get(SRC, "stencil_loss", Dict{String,String}())
SRC["stencil_loss"]["cuda_f64"] = raw"""using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_stencil_loss_b_1!(i_n, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function cuda_kernel_stencil_loss_b_2!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x2 = 2 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_x2] ^ 2
    return nothing
end

function cuda_kernel_stencil_loss_b_3!(i_n, lossb, w, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(2 - (i_n - 1), -1) + 1
        return nothing
    end
    i_x2 = (i_n - 1) + (__tid - 1) * -1
    wb[i_x2] = wb[i_x2] + (2 * w[i_x2]) * lossb[1]
    return nothing
end

function cuda_kernel_stencil_loss_b_4!(i_n, ub, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(2 - (i_n - 1), -1) + 1
        return nothing
    end
    i_x = (i_n - 1) + (__tid - 1) * -1
    __oldb_0 = wb[i_x]
    wb[i_x] = 0.0
    ub[i_x - 1] = ub[i_x - 1] + __oldb_0
    ub[i_x] = ub[i_x] + 2.0 * -__oldb_0
    ub[i_x + 1] = ub[i_x + 1] + __oldb_0
    return nothing
end

function cuda_kernel_stencil_loss_1!(i_n, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function cuda_kernel_stencil_loss_2!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x2 = 2 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_x2] ^ 2
    return nothing
end

function initstacks_stencil_loss_b_cuda()
    return nothing
end

function stencil_loss_b_cuda(loss, lossb, u, ub, w, wb, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_b_1!(i_n, u, w)
    if div((i_n - 1) - 2, 1) + 1 < 32768
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_b_2!(i_n, loss, w)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(w, 2:i_n - 1))), abs2, view(w, 2:i_n - 1))
            end
    end
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(2 - (i_n - 1), -1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_b_3!(i_n, lossb, w, wb)
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(2 - (i_n - 1), -1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_b_4!(i_n, ub, wb)
    return nothing
end

function stencil_loss_cuda(loss, u, w, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_1!(i_n, u, w)
    if div((i_n - 1) - 2, 1) + 1 < 32768
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_2!(i_n, loss, w)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(w, 2:i_n - 1))), abs2, view(w, 2:i_n - 1))
            end
    end
    return nothing
end
"""
SRC["stencil_loss"] = get(SRC, "stencil_loss", Dict{String,String}())
SRC["stencil_loss"]["cuda_f32"] = raw"""using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_stencil_loss_b_1!(i_n, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    w[i_x] = (u[i_x - 1] - 2.0f0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function cuda_kernel_stencil_loss_b_2!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x2 = 2 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_x2] ^ 2
    return nothing
end

function cuda_kernel_stencil_loss_b_3!(i_n, lossb, w, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(2 - (i_n - 1), -1) + 1
        return nothing
    end
    i_x2 = (i_n - 1) + (__tid - 1) * -1
    wb[i_x2] = wb[i_x2] + (2 * w[i_x2]) * lossb[1]
    return nothing
end

function cuda_kernel_stencil_loss_b_4!(i_n, ub, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(2 - (i_n - 1), -1) + 1
        return nothing
    end
    i_x = (i_n - 1) + (__tid - 1) * -1
    __oldb_0 = wb[i_x]
    wb[i_x] = 0.0f0
    ub[i_x - 1] = ub[i_x - 1] + __oldb_0
    ub[i_x] = ub[i_x] + 2.0f0 * -__oldb_0
    ub[i_x + 1] = ub[i_x + 1] + __oldb_0
    return nothing
end

function cuda_kernel_stencil_loss_1!(i_n, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    w[i_x] = (u[i_x - 1] - 2.0f0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function cuda_kernel_stencil_loss_2!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x2 = 2 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_x2] ^ 2
    return nothing
end

function initstacks_stencil_loss_b_cuda()
    return nothing
end

function stencil_loss_b_cuda(loss, lossb, u, ub, w, wb, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_b_1!(i_n, u, w)
    if div((i_n - 1) - 2, 1) + 1 < 32768
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_b_2!(i_n, loss, w)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(w, 2:i_n - 1))), abs2, view(w, 2:i_n - 1))
            end
    end
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(2 - (i_n - 1), -1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_b_3!(i_n, lossb, w, wb)
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(2 - (i_n - 1), -1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_b_4!(i_n, ub, wb)
    return nothing
end

function stencil_loss_cuda(loss, u, w, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_1!(i_n, u, w)
    if div((i_n - 1) - 2, 1) + 1 < 32768
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block)) cuda_kernel_stencil_loss_2!(i_n, loss, w)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(w, 2:i_n - 1))), abs2, view(w, 2:i_n - 1))
            end
    end
    return nothing
end
"""
SRC["stencil_loss"] = get(SRC, "stencil_loss", Dict{String,String}())
SRC["stencil_loss"]["jacc_f64"] = raw"""using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_stencil_loss_b_1!(__jacc_i, i_n, u, w)
    i_x = 2 + (__jacc_i - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function jacc_kernel_stencil_loss_b_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_stencil_loss_b_3!(__jacc_i, i_n, loss, w)
    i_x2 = 2 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_x2] ^ 2
    return nothing
end

function jacc_kernel_stencil_loss_b_4!(__jacc_i, i_n, lossb, w, wb)
    i_x2 = (i_n - 1) + (__jacc_i - 1) * -1
    wb[i_x2] = wb[i_x2] + (2 * w[i_x2]) * lossb[1]
    return nothing
end

function jacc_kernel_stencil_loss_b_5!(__jacc_i, i_n, ub, wb)
    i_x = (i_n - 1) + (__jacc_i - 1) * -1
    __oldb_0 = wb[i_x]
    wb[i_x] = 0.0
    ub[i_x - 1] = ub[i_x - 1] + __oldb_0
    ub[i_x] = ub[i_x] + 2.0 * -__oldb_0
    ub[i_x + 1] = ub[i_x + 1] + __oldb_0
    return nothing
end

function jacc_kernel_stencil_loss_1!(__jacc_i, i_n, u, w)
    i_x = 2 + (__jacc_i - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function jacc_kernel_stencil_loss_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_stencil_loss_3!(__jacc_i, i_n, loss, w)
    i_x2 = 2 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_x2] ^ 2
    return nothing
end

function initstacks_stencil_loss_b_jacc()
    return nothing
end

function stencil_loss_b_jacc(loss, lossb, u, ub, w, wb, i_n)
    if div((i_n - 1) - 2, 1) + 1 > 0
        JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_b_1!(i_n, u, w)
    end
    if div((i_n - 1) - 2, 1) + 1 < 32768
        if div((i_n - 1) - 2, 1) + 1 > 0
            JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_b_3!(i_n, loss, w)
        end
    else
        __jgen_redval_2 = JACC.@parallel_reduce(range = div((i_n - 1) - 2, 1) + 1, (((i_x2, w)->w[i_x2] ^ 2))(w))
        JACC.@parallel_for range = 1 jacc_kernel_stencil_loss_b_2!(loss, __jgen_redval_2)
    end
    if div(2 - (i_n - 1), -1) + 1 > 0
        JACC.@parallel_for range = div(2 - (i_n - 1), -1) + 1 jacc_kernel_stencil_loss_b_4!(i_n, lossb, w, wb)
    end
    if div(2 - (i_n - 1), -1) + 1 > 0
        JACC.@parallel_for range = div(2 - (i_n - 1), -1) + 1 jacc_kernel_stencil_loss_b_5!(i_n, ub, wb)
    end
    return nothing
end

function stencil_loss_jacc(loss, u, w, i_n)
    if div((i_n - 1) - 2, 1) + 1 > 0
        JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_1!(i_n, u, w)
    end
    if div((i_n - 1) - 2, 1) + 1 < 32768
        if div((i_n - 1) - 2, 1) + 1 > 0
            JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_3!(i_n, loss, w)
        end
    else
        __jgen_redval_2 = JACC.@parallel_reduce(range = div((i_n - 1) - 2, 1) + 1, (((i_x2, w)->w[i_x2] ^ 2))(w))
        JACC.@parallel_for range = 1 jacc_kernel_stencil_loss_2!(loss, __jgen_redval_2)
    end
    return nothing
end
"""
SRC["stencil_loss"] = get(SRC, "stencil_loss", Dict{String,String}())
SRC["stencil_loss"]["jacc_f32"] = raw"""using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_stencil_loss_b_1!(__jacc_i, i_n, u, w)
    i_x = 2 + (__jacc_i - 1)
    w[i_x] = (u[i_x - 1] - 2.0f0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function jacc_kernel_stencil_loss_b_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_stencil_loss_b_3!(__jacc_i, i_n, loss, w)
    i_x2 = 2 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_x2] ^ 2
    return nothing
end

function jacc_kernel_stencil_loss_b_4!(__jacc_i, i_n, lossb, w, wb)
    i_x2 = (i_n - 1) + (__jacc_i - 1) * -1
    wb[i_x2] = wb[i_x2] + (2 * w[i_x2]) * lossb[1]
    return nothing
end

function jacc_kernel_stencil_loss_b_5!(__jacc_i, i_n, ub, wb)
    i_x = (i_n - 1) + (__jacc_i - 1) * -1
    __oldb_0 = wb[i_x]
    wb[i_x] = 0.0f0
    ub[i_x - 1] = ub[i_x - 1] + __oldb_0
    ub[i_x] = ub[i_x] + 2.0f0 * -__oldb_0
    ub[i_x + 1] = ub[i_x + 1] + __oldb_0
    return nothing
end

function jacc_kernel_stencil_loss_1!(__jacc_i, i_n, u, w)
    i_x = 2 + (__jacc_i - 1)
    w[i_x] = (u[i_x - 1] - 2.0f0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function jacc_kernel_stencil_loss_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_stencil_loss_3!(__jacc_i, i_n, loss, w)
    i_x2 = 2 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_x2] ^ 2
    return nothing
end

function initstacks_stencil_loss_b_jacc()
    return nothing
end

function stencil_loss_b_jacc(loss, lossb, u, ub, w, wb, i_n)
    if div((i_n - 1) - 2, 1) + 1 > 0
        JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_b_1!(i_n, u, w)
    end
    if div((i_n - 1) - 2, 1) + 1 < 32768
        if div((i_n - 1) - 2, 1) + 1 > 0
            JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_b_3!(i_n, loss, w)
        end
    else
        __jgen_redval_2 = JACC.@parallel_reduce(range = div((i_n - 1) - 2, 1) + 1, (((i_x2, w)->w[i_x2] ^ 2))(w))
        JACC.@parallel_for range = 1 jacc_kernel_stencil_loss_b_2!(loss, __jgen_redval_2)
    end
    if div(2 - (i_n - 1), -1) + 1 > 0
        JACC.@parallel_for range = div(2 - (i_n - 1), -1) + 1 jacc_kernel_stencil_loss_b_4!(i_n, lossb, w, wb)
    end
    if div(2 - (i_n - 1), -1) + 1 > 0
        JACC.@parallel_for range = div(2 - (i_n - 1), -1) + 1 jacc_kernel_stencil_loss_b_5!(i_n, ub, wb)
    end
    return nothing
end

function stencil_loss_jacc(loss, u, w, i_n)
    if div((i_n - 1) - 2, 1) + 1 > 0
        JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_1!(i_n, u, w)
    end
    if div((i_n - 1) - 2, 1) + 1 < 32768
        if div((i_n - 1) - 2, 1) + 1 > 0
            JACC.@parallel_for range = div((i_n - 1) - 2, 1) + 1 jacc_kernel_stencil_loss_3!(i_n, loss, w)
        end
    else
        __jgen_redval_2 = JACC.@parallel_reduce(range = div((i_n - 1) - 2, 1) + 1, (((i_x2, w)->w[i_x2] ^ 2))(w))
        JACC.@parallel_for range = 1 jacc_kernel_stencil_loss_2!(loss, __jgen_redval_2)
    end
    return nothing
end
"""
SRC["matvec_loss"] = get(SRC, "matvec_loss", Dict{String,String}())
SRC["matvec_loss"]["cuda_f64"] = raw"""using CUDA
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
"""
SRC["matvec_loss"] = get(SRC, "matvec_loss", Dict{String,String}())
SRC["matvec_loss"]["cuda_f32"] = raw"""using CUDA
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
"""
SRC["matvec_loss"] = get(SRC, "matvec_loss", Dict{String,String}())
SRC["matvec_loss"]["jacc_f64"] = raw"""using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_matvec_loss_b_1!(__jacc_i, a, i_m, i_n, u, v)
    idx = 1 + (__jacc_i - 1)
    __icse_0 = idx - 1
    i_i = div(__icse_0, i_n) + 1
    i_j = mod(__icse_0, i_n) + 1
    Atomix.@atomic v[i_i] += a[i_i, i_j] * u[i_j]
    return nothing
end

function jacc_kernel_matvec_loss_b_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_matvec_loss_b_3!(__jacc_i, i_m, loss, v)
    i_i2 = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += v[i_i2] ^ 2
    return nothing
end

function jacc_kernel_matvec_loss_b_4!(__jacc_i, i_m, lossb, v, vb)
    i_i2 = i_m + (__jacc_i - 1) * -1
    vb[i_i2] = vb[i_i2] + (2 * v[i_i2]) * lossb[1]
    return nothing
end

function jacc_kernel_matvec_loss_b_5!(__jacc_i, a, ab, i_m, i_n, u, ub, vb)
    idx = i_m * i_n + (__jacc_i - 1) * -1
    __icse_1 = idx - 1
    i_i = div(__icse_1, i_n) + 1
    i_j = mod(__icse_1, i_n) + 1
    __cse_2 = vb[i_i]
    Atomix.@atomic ab[i_i, i_j] += u[i_j] * __cse_2
    Atomix.@atomic ub[i_j] += a[i_i, i_j] * __cse_2
    return nothing
end

function jacc_kernel_matvec_loss_1!(__jacc_i, a, i_m, i_n, u, v)
    idx = 1 + (__jacc_i - 1)
    i_i = div(idx - 1, i_n) + 1
    i_j = mod(idx - 1, i_n) + 1
    Atomix.@atomic v[i_i] += a[i_i, i_j] * u[i_j]
    return nothing
end

function jacc_kernel_matvec_loss_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_matvec_loss_3!(__jacc_i, i_m, loss, v)
    i_i2 = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += v[i_i2] ^ 2
    return nothing
end

function initstacks_matvec_loss_b_jacc()
    return nothing
end

function matvec_loss_b_jacc(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n)
    if div(i_m * i_n - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(i_m * i_n - 1, 1) + 1 jacc_kernel_matvec_loss_b_1!(a, i_m, i_n, u, v)
    end
    if div(i_m - 1, 1) + 1 < 32768
        if div(i_m - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_b_3!(i_m, loss, v)
        end
    else
        __jgen_redval_2 = JACC.@parallel_reduce(range = div(i_m - 1, 1) + 1, (((i_i2, v)->v[i_i2] ^ 2))(v))
        JACC.@parallel_for range = 1 jacc_kernel_matvec_loss_b_2!(loss, __jgen_redval_2)
    end
    if div(1 - i_m, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - i_m, -1) + 1 jacc_kernel_matvec_loss_b_4!(i_m, lossb, v, vb)
    end
    if div(1 - i_m * i_n, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - i_m * i_n, -1) + 1 jacc_kernel_matvec_loss_b_5!(a, ab, i_m, i_n, u, ub, vb)
    end
    return nothing
end

function matvec_loss_jacc(loss, a, u, v, i_m, i_n)
    if div(i_m * i_n - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(i_m * i_n - 1, 1) + 1 jacc_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    end
    if div(i_m - 1, 1) + 1 < 32768
        if div(i_m - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_3!(i_m, loss, v)
        end
    else
        __jgen_redval_2 = JACC.@parallel_reduce(range = div(i_m - 1, 1) + 1, (((i_i2, v)->v[i_i2] ^ 2))(v))
        JACC.@parallel_for range = 1 jacc_kernel_matvec_loss_2!(loss, __jgen_redval_2)
    end
    return nothing
end
"""
SRC["matvec_loss"] = get(SRC, "matvec_loss", Dict{String,String}())
SRC["matvec_loss"]["jacc_f32"] = raw"""using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_matvec_loss_b_1!(__jacc_i, a, i_m, i_n, u, v)
    idx = 1 + (__jacc_i - 1)
    __icse_0 = idx - 1
    i_i = div(__icse_0, i_n) + 1
    i_j = mod(__icse_0, i_n) + 1
    Atomix.@atomic v[i_i] += a[i_i, i_j] * u[i_j]
    return nothing
end

function jacc_kernel_matvec_loss_b_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_matvec_loss_b_3!(__jacc_i, i_m, loss, v)
    i_i2 = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += v[i_i2] ^ 2
    return nothing
end

function jacc_kernel_matvec_loss_b_4!(__jacc_i, i_m, lossb, v, vb)
    i_i2 = i_m + (__jacc_i - 1) * -1
    vb[i_i2] = vb[i_i2] + (2 * v[i_i2]) * lossb[1]
    return nothing
end

function jacc_kernel_matvec_loss_b_5!(__jacc_i, a, ab, i_m, i_n, u, ub, vb)
    idx = i_m * i_n + (__jacc_i - 1) * -1
    __icse_1 = idx - 1
    i_i = div(__icse_1, i_n) + 1
    i_j = mod(__icse_1, i_n) + 1
    __cse_2 = vb[i_i]
    Atomix.@atomic ab[i_i, i_j] += u[i_j] * __cse_2
    Atomix.@atomic ub[i_j] += a[i_i, i_j] * __cse_2
    return nothing
end

function jacc_kernel_matvec_loss_1!(__jacc_i, a, i_m, i_n, u, v)
    idx = 1 + (__jacc_i - 1)
    i_i = div(idx - 1, i_n) + 1
    i_j = mod(idx - 1, i_n) + 1
    Atomix.@atomic v[i_i] += a[i_i, i_j] * u[i_j]
    return nothing
end

function jacc_kernel_matvec_loss_2!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_matvec_loss_3!(__jacc_i, i_m, loss, v)
    i_i2 = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += v[i_i2] ^ 2
    return nothing
end

function initstacks_matvec_loss_b_jacc()
    return nothing
end

function matvec_loss_b_jacc(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n)
    if div(i_m * i_n - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(i_m * i_n - 1, 1) + 1 jacc_kernel_matvec_loss_b_1!(a, i_m, i_n, u, v)
    end
    if div(i_m - 1, 1) + 1 < 32768
        if div(i_m - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_b_3!(i_m, loss, v)
        end
    else
        __jgen_redval_2 = JACC.@parallel_reduce(range = div(i_m - 1, 1) + 1, (((i_i2, v)->v[i_i2] ^ 2))(v))
        JACC.@parallel_for range = 1 jacc_kernel_matvec_loss_b_2!(loss, __jgen_redval_2)
    end
    if div(1 - i_m, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - i_m, -1) + 1 jacc_kernel_matvec_loss_b_4!(i_m, lossb, v, vb)
    end
    if div(1 - i_m * i_n, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - i_m * i_n, -1) + 1 jacc_kernel_matvec_loss_b_5!(a, ab, i_m, i_n, u, ub, vb)
    end
    return nothing
end

function matvec_loss_jacc(loss, a, u, v, i_m, i_n)
    if div(i_m * i_n - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(i_m * i_n - 1, 1) + 1 jacc_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    end
    if div(i_m - 1, 1) + 1 < 32768
        if div(i_m - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_3!(i_m, loss, v)
        end
    else
        __jgen_redval_2 = JACC.@parallel_reduce(range = div(i_m - 1, 1) + 1, (((i_i2, v)->v[i_i2] ^ 2))(v))
        JACC.@parallel_for range = 1 jacc_kernel_matvec_loss_2!(loss, __jgen_redval_2)
    end
    return nothing
end
"""
SRC["mlp1d"] = get(SRC, "mlp1d", Dict{String,String}())
SRC["mlp1d"]["cuda_f64"] = raw"""using CUDA
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
"""
SRC["mlp1d"] = get(SRC, "mlp1d", Dict{String,String}())
SRC["mlp1d"]["cuda_f32"] = raw"""using CUDA
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
    s2b = 0.0f0
    s2 = b2[i_j2]
    for i_k = 1:n_h
        s2 = s2 + w2[(i_j2 - 1) * n_h + i_k] * h1[i_k]
    end
    __oldb_0 = h2b[i_j2]
    h2b[i_j2] = 0.0f0
    s2b = s2b + (1.0f0 - tanh(s2) ^ 2) * __oldb_0
    for i_k = 1:n_h
        __cse_2 = w2[(i_j2 - 1) * n_h + i_k]
        __cse_3 = h1[i_k]
        s2 = s2 + __cse_2 * __cse_3
        w2b[(i_j2 - 1) * n_h + i_k] = w2b[(i_j2 - 1) * n_h + i_k] + __cse_3 * s2b
        CUDA.@atomic h1b[i_k] += __cse_2 * s2b
    end
    __oldb_0 = s2b
    s2b = 0.0f0
    b2b[i_j2] = b2b[i_j2] + __oldb_0
    return nothing
end

function cuda_kernel_mlp1d_b_5!(b1, b1b, h1b, n_h, w1, w1b, x, xb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_h - 1, 1) + 1
        return nothing
    end
    i_j = 1 + (__tid - 1)
    s1b = 0.0f0
    __cse_4 = w1[i_j]
    __cse_5 = x[1]
    s1 = __cse_4 * __cse_5 + b1[i_j]
    __oldb_0 = h1b[i_j]
    h1b[i_j] = 0.0f0
    s1b = s1b + (1.0f0 - tanh(s1) ^ 2) * __oldb_0
    __oldb_0 = s1b
    s1b = 0.0f0
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
    s1 = 0.0f0
    s2 = 0.0f0
    s3 = 0.0f0
    s1b = 0.0f0
    s2b = 0.0f0
    s3b = 0.0f0
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
            ob[1] = 0.0f0
            s3b = s3b + __oldb_0
        end
    @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(1 - n_h, -1) + 1, nthread_per_block)) cuda_kernel_mlp1d_b_3!(h2, h2b, n_h, s3b, w3, w3b)
    CUDA.@allowscalar begin
            __oldb_0 = s3b
            s3b = 0.0f0
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
"""
SRC["mlp1d"] = get(SRC, "mlp1d", Dict{String,String}())
SRC["mlp1d"]["jacc_f64"] = raw"""using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_mlp1d_b_1!(__jacc_i, b1, h1, n_h, w1, x)
    i_j = 1 + (__jacc_i - 1)
    s1 = w1[i_j] * x[1] + b1[i_j]
    h1[i_j] = tanh(s1)
    return nothing
end

function jacc_kernel_mlp1d_b_2!(__jacc_i, b2, h1, h2, n_h, w2)
    i_j2 = 1 + (__jacc_i - 1)
    s2 = b2[i_j2]
    for i_k = 1:n_h
        s2 = s2 + w2[(i_j2 - 1) * n_h + i_k] * h1[i_k]
    end
    h2[i_j2] = tanh(s2)
    return nothing
end

function jacc_kernel_mlp1d_b_3!(__jacc_i, h2, h2b, n_h, s3b, w3, w3b)
    i_k2 = n_h + (__jacc_i - 1) * -1
    w3b[i_k2] = w3b[i_k2] + h2[i_k2] * s3b
    h2b[i_k2] = h2b[i_k2] + w3[i_k2] * s3b
    return nothing
end

function jacc_kernel_mlp1d_b_4!(__jacc_i, b2, b2b, h1, h1b, h2b, n_h, w2, w2b)
    i_j2 = 1 + (__jacc_i - 1)
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
        Atomix.@atomic h1b[i_k] += __cse_2 * s2b
    end
    __oldb_0 = s2b
    s2b = 0.0
    b2b[i_j2] = b2b[i_j2] + __oldb_0
    return nothing
end

function jacc_kernel_mlp1d_b_5!(__jacc_i, b1, b1b, h1b, n_h, w1, w1b, x, xb)
    i_j = 1 + (__jacc_i - 1)
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
    Atomix.@atomic xb[1] += __cse_4 * __oldb_0
    b1b[i_j] = b1b[i_j] + __oldb_0
    return nothing
end

function jacc_kernel_mlp1d_1!(__jacc_i, b1, h1, n_h, w1, x)
    i_j = 1 + (__jacc_i - 1)
    s1 = w1[i_j] * x[1] + b1[i_j]
    h1[i_j] = tanh(s1)
    return nothing
end

function jacc_kernel_mlp1d_2!(__jacc_i, b2, h1, h2, n_h, w2)
    i_j2 = 1 + (__jacc_i - 1)
    s2 = b2[i_j2]
    for i_k = 1:n_h
        s2 = s2 + w2[(i_j2 - 1) * n_h + i_k] * h1[i_k]
    end
    h2[i_j2] = tanh(s2)
    return nothing
end

function initstacks_mlp1d_b_jacc(n_h)
    return nothing
end

function mlp1d_b_jacc(loss, lossb, x, xb, y, yb, w1, w1b, b1, b1b, w2, w2b, b2, b2b, w3, w3b, b3, b3b, h1, h1b, h2, h2b, o, ob, n_h)
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s1b = 0.0
    s2b = 0.0
    s3b = 0.0
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_b_1!(b1, h1, n_h, w1, x)
    end
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_b_2!(b2, h1, h2, n_h, w2)
    end
    CUDA.@allowscalar begin
            s3 = b3[1]
        end
    for i_k2 = 1:n_h
        CUDA.@allowscalar begin
                s3 = s3 + w3[i_k2] * h2[i_k2]
            end
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
    if div(1 - n_h, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_h, -1) + 1 jacc_kernel_mlp1d_b_3!(h2, h2b, n_h, s3b, w3, w3b)
    end
    CUDA.@allowscalar begin
            __oldb_0 = s3b
            s3b = 0.0
            b3b[1] = b3b[1] + __oldb_0
        end
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_b_4!(b2, b2b, h1, h1b, h2b, n_h, w2, w2b)
    end
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_b_5!(b1, b1b, h1b, n_h, w1, w1b, x, xb)
    end
    return nothing
end

function mlp1d_jacc(loss, x, y, w1, b1, w2, b2, w3, b3, h1, h2, o, n_h)
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_1!(b1, h1, n_h, w1, x)
    end
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_2!(b2, h1, h2, n_h, w2)
    end
    CUDA.@allowscalar begin
            s3 = b3[1]
        end
    for i_k2 = 1:n_h
        CUDA.@allowscalar begin
                s3 = s3 + w3[i_k2] * h2[i_k2]
            end
    end
    CUDA.@allowscalar begin
            o[1] = s3
            loss[1] = loss[1] + (o[1] - y[1]) ^ 2
        end
    return nothing
end
"""
SRC["mlp1d"] = get(SRC, "mlp1d", Dict{String,String}())
SRC["mlp1d"]["jacc_f32"] = raw"""using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_mlp1d_b_1!(__jacc_i, b1, h1, n_h, w1, x)
    i_j = 1 + (__jacc_i - 1)
    s1 = w1[i_j] * x[1] + b1[i_j]
    h1[i_j] = tanh(s1)
    return nothing
end

function jacc_kernel_mlp1d_b_2!(__jacc_i, b2, h1, h2, n_h, w2)
    i_j2 = 1 + (__jacc_i - 1)
    s2 = b2[i_j2]
    for i_k = 1:n_h
        s2 = s2 + w2[(i_j2 - 1) * n_h + i_k] * h1[i_k]
    end
    h2[i_j2] = tanh(s2)
    return nothing
end

function jacc_kernel_mlp1d_b_3!(__jacc_i, h2, h2b, n_h, s3b, w3, w3b)
    i_k2 = n_h + (__jacc_i - 1) * -1
    w3b[i_k2] = w3b[i_k2] + h2[i_k2] * s3b
    h2b[i_k2] = h2b[i_k2] + w3[i_k2] * s3b
    return nothing
end

function jacc_kernel_mlp1d_b_4!(__jacc_i, b2, b2b, h1, h1b, h2b, n_h, w2, w2b)
    i_j2 = 1 + (__jacc_i - 1)
    s2b = 0.0f0
    s2 = b2[i_j2]
    for i_k = 1:n_h
        s2 = s2 + w2[(i_j2 - 1) * n_h + i_k] * h1[i_k]
    end
    __oldb_0 = h2b[i_j2]
    h2b[i_j2] = 0.0f0
    s2b = s2b + (1.0f0 - tanh(s2) ^ 2) * __oldb_0
    for i_k = 1:n_h
        __cse_2 = w2[(i_j2 - 1) * n_h + i_k]
        __cse_3 = h1[i_k]
        s2 = s2 + __cse_2 * __cse_3
        w2b[(i_j2 - 1) * n_h + i_k] = w2b[(i_j2 - 1) * n_h + i_k] + __cse_3 * s2b
        Atomix.@atomic h1b[i_k] += __cse_2 * s2b
    end
    __oldb_0 = s2b
    s2b = 0.0f0
    b2b[i_j2] = b2b[i_j2] + __oldb_0
    return nothing
end

function jacc_kernel_mlp1d_b_5!(__jacc_i, b1, b1b, h1b, n_h, w1, w1b, x, xb)
    i_j = 1 + (__jacc_i - 1)
    s1b = 0.0f0
    __cse_4 = w1[i_j]
    __cse_5 = x[1]
    s1 = __cse_4 * __cse_5 + b1[i_j]
    __oldb_0 = h1b[i_j]
    h1b[i_j] = 0.0f0
    s1b = s1b + (1.0f0 - tanh(s1) ^ 2) * __oldb_0
    __oldb_0 = s1b
    s1b = 0.0f0
    w1b[i_j] = w1b[i_j] + __cse_5 * __oldb_0
    Atomix.@atomic xb[1] += __cse_4 * __oldb_0
    b1b[i_j] = b1b[i_j] + __oldb_0
    return nothing
end

function jacc_kernel_mlp1d_1!(__jacc_i, b1, h1, n_h, w1, x)
    i_j = 1 + (__jacc_i - 1)
    s1 = w1[i_j] * x[1] + b1[i_j]
    h1[i_j] = tanh(s1)
    return nothing
end

function jacc_kernel_mlp1d_2!(__jacc_i, b2, h1, h2, n_h, w2)
    i_j2 = 1 + (__jacc_i - 1)
    s2 = b2[i_j2]
    for i_k = 1:n_h
        s2 = s2 + w2[(i_j2 - 1) * n_h + i_k] * h1[i_k]
    end
    h2[i_j2] = tanh(s2)
    return nothing
end

function initstacks_mlp1d_b_jacc(n_h)
    return nothing
end

function mlp1d_b_jacc(loss, lossb, x, xb, y, yb, w1, w1b, b1, b1b, w2, w2b, b2, b2b, w3, w3b, b3, b3b, h1, h1b, h2, h2b, o, ob, n_h)
    s1 = 0.0f0
    s2 = 0.0f0
    s3 = 0.0f0
    s1b = 0.0f0
    s2b = 0.0f0
    s3b = 0.0f0
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_b_1!(b1, h1, n_h, w1, x)
    end
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_b_2!(b2, h1, h2, n_h, w2)
    end
    CUDA.@allowscalar begin
            s3 = b3[1]
        end
    for i_k2 = 1:n_h
        CUDA.@allowscalar begin
                s3 = s3 + w3[i_k2] * h2[i_k2]
            end
    end
    CUDA.@allowscalar begin
            o[1] = s3
            __cse_0 = o[1] - y[1]
            loss[1] = loss[1] + __cse_0 ^ 2
            __cse_1 = (2__cse_0) * lossb[1]
            ob[1] = ob[1] + __cse_1
            yb[1] = yb[1] + -__cse_1
            __oldb_0 = ob[1]
            ob[1] = 0.0f0
            s3b = s3b + __oldb_0
        end
    if div(1 - n_h, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - n_h, -1) + 1 jacc_kernel_mlp1d_b_3!(h2, h2b, n_h, s3b, w3, w3b)
    end
    CUDA.@allowscalar begin
            __oldb_0 = s3b
            s3b = 0.0f0
            b3b[1] = b3b[1] + __oldb_0
        end
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_b_4!(b2, b2b, h1, h1b, h2b, n_h, w2, w2b)
    end
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_b_5!(b1, b1b, h1b, n_h, w1, w1b, x, xb)
    end
    return nothing
end

function mlp1d_jacc(loss, x, y, w1, b1, w2, b2, w3, b3, h1, h2, o, n_h)
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_1!(b1, h1, n_h, w1, x)
    end
    if div(n_h - 1, 1) + 1 > 0
        JACC.@parallel_for range = div(n_h - 1, 1) + 1 jacc_kernel_mlp1d_2!(b2, h1, h2, n_h, w2)
    end
    CUDA.@allowscalar begin
            s3 = b3[1]
        end
    for i_k2 = 1:n_h
        CUDA.@allowscalar begin
                s3 = s3 + w3[i_k2] * h2[i_k2]
            end
    end
    CUDA.@allowscalar begin
            o[1] = s3
            loss[1] = loss[1] + (o[1] - y[1]) ^ 2
        end
    return nothing
end
"""

const PY_SCRIPT = raw"""# Single-precision health check harness. Modes: torch | jax64 | jax32
# Reads inputs written by Julia, computes gradients, compares with STADE outputs and a float64 reference.
import sys, os, json, traceback
import numpy as np
mode, d = sys.argv[1], sys.argv[2]
meta = json.load(open(d + "/meta.json"))
DIFF = {"stencil_loss": ["u"], "matvec_loss": ["a", "u"],
        "mlp1d": ["x", "y", "w1", "b1", "w2", "b2", "w3", "b3"]}
DIFF["stencil_loss_big"] = DIFF["stencil_loss"]
def base(k): return k.replace("_big", "")
def load(path, shape):
    return np.fromfile(path, dtype="<f8").reshape(shape, order="F")
def rel(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
def inputs(k):
    return {n: load(f"{d}/in_{k}_{n}.bin", s) for n, s in meta["shapes"][k].items()}
def errs(k, loss, grads, ref_loss, ref_grads):
    e = {n: rel(grads[n], ref_grads[n]) for n in DIFF[k]}
    e_loss = abs(loss - ref_loss) / max(abs(ref_loss), 1e-300)
    return {"loss_rel_err": float(e_loss), "grad_rel_err": e, "max_rel_err": float(max([e_loss] + list(e.values())))}

out = {"mode": mode}
if mode == "torch":
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out["torch"] = torch.__version__; out["device"] = torch.cuda.get_device_name(0)
    def loss_t(k, T):
        k = base(k)
        if k == "stencil_loss":
            u = T["u"]; w = u[:-2] - 2.0 * u[1:-1] + u[2:]; return (w * w).sum()
        if k == "matvec_loss":
            v = T["a"] @ T["u"]; return (v * v).sum()
        nh = meta["ints"]["mlp1d"]["n_h"]
        h1 = torch.tanh(T["w1"] * T["x"][0] + T["b1"])
        h2 = torch.tanh(T["w2"].reshape(nh, nh) @ h1 + T["b2"])
        o = (T["w3"] * h2).sum() + T["b3"][0]
        return (o - T["y"][0]) ** 2
    def run(k, dt):
        I = inputs(k)
        T = {n: torch.tensor(a, dtype=dt, device="cuda", requires_grad=(n in DIFF[k])) for n, a in I.items()}
        L = loss_t(k, T); L.backward(); torch.cuda.synchronize()
        return float(L), {n: T[n].grad.double().cpu().numpy() for n in DIFF[k]}, str(T[DIFF[k][0]].grad.dtype)
    for k in meta["kernels"]:
        try:
            ref_l, ref_g, _ = run(k, torch.float64)
            for n, g in ref_g.items(): np.asarray(g).flatten(order="F").astype("<f8").tofile(f"{d}/ref_{k}_{n}.bin")  # column-major, like Julia
            json.dump({"loss": ref_l}, open(f"{d}/ref_{k}_loss.json", "w"))
            r = {"torch_f64": dict(errs(k, ref_l, ref_g, ref_l, ref_g), note="reference")}
            l32, g32, dt32 = run(k, torch.float32)
            r["torch_f32"] = dict(errs(k, l32, g32, ref_l, ref_g), grad_dtype=dt32)
            for var in meta["variants"]:
                lp = f"{d}/out_{k}_{var}_loss.bin"
                if not os.path.exists(lp): continue
                sl = float(np.fromfile(lp, dtype="<f8")[0])
                sg = {n: load(f"{d}/out_{k}_{var}_{n}.bin", ref_g[n].shape) for n in DIFF[k]}
                r["stade_" + var] = errs(k, sl, sg, ref_l, ref_g)
            out[k] = r
        except Exception as ex:
            out[k] = {"error": repr(ex)[:400], "trace": traceback.format_exc()[-700:]}
else:
    import jax
    if mode == "jax64": jax.config.update("jax_enable_x64", True)
    else: jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    out["jax"] = jax.__version__; out["backend"] = jax.default_backend(); out["x64"] = bool(jax.config.jax_enable_x64)
    if out["backend"] not in ("gpu", "cuda") and os.environ.get("SP_ALLOW_CPU") != "1":
        raise RuntimeError("JAX is not on the GPU: " + out["backend"])
    dt = jnp.float64 if mode == "jax64" else jnp.float32
    def loss_j(k, T):
        k = base(k)
        if k == "stencil_loss":
            u = T["u"]; w = u[:-2] - 2.0 * u[1:-1] + u[2:]; return jnp.sum(w * w)
        if k == "matvec_loss":
            v = T["a"] @ T["u"]; return jnp.sum(v * v)
        nh = meta["ints"]["mlp1d"]["n_h"]
        h1 = jnp.tanh(T["w1"] * T["x"][0] + T["b1"])
        h2 = jnp.tanh(T["w2"].reshape(nh, nh) @ h1 + T["b2"])
        o = jnp.sum(T["w3"] * h2) + T["b3"][0]
        return (o - T["y"][0]) ** 2
    for k in meta["kernels"]:
        try:
            I = {n: jnp.asarray(a, dtype=dt) for n, a in inputs(k).items()}
            diff = {n: I[n] for n in DIFF[k]}; rest = {n: v for n, v in I.items() if n not in diff}
            f = jax.jit(jax.value_and_grad(lambda D: loss_j(k, {**D, **rest})))
            L, G = f(diff); L.block_until_ready()
            ref_l = json.load(open(f"{d}/ref_{k}_loss.json"))["loss"]
            ref_g = {n: load(f"{d}/ref_{k}_{n}.bin", np.asarray(diff[n]).shape) for n in DIFF[k]}
            out[k] = {"jax_" + mode[3:]: dict(errs(k, float(L), {n: np.asarray(G[n]) for n in DIFF[k]}, ref_l, ref_g),
                                              grad_dtype=str(G[DIFF[k][0]].dtype))}
        except Exception as ex:
            out[k] = {"error": repr(ex)[:400], "trace": traceback.format_exc()[-700:]}
print(json.dumps(out))
"""
const JAX_LAUNCH = raw"""source /scratch/coop/drozda/jax-env/bin/activate
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(ls -d $SP/nvidia/*/lib | tr '\n' ':')"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python "$@"
"""
const TORCH_LAUNCH = raw"""source /scratch/coop/drozda/torch-env/bin/activate
python "$@"
"""

fgen(n, a, b, c) = [sin(a * i + b) + c * cos(0.7 * i) for i in 1:n]

# case name => (kernel function, spec in primal argument order, integer names for initstacks)
const SPECS = Dict(
    "stencil_loss" => ([("loss", :f), ("u", :f), ("w", :f), ("i_n", :i)], String[]),
    "matvec_loss"  => ([("loss", :f), ("a", :f), ("u", :f), ("v", :f), ("i_m", :i), ("i_n", :i)], String[]),
    "mlp1d"        => ([("loss", :f), ("x", :f), ("y", :f), ("w1", :f), ("b1", :f), ("w2", :f), ("b2", :f),
                        ("w3", :f), ("b3", :f), ("h1", :f), ("h2", :f), ("o", :f), ("n_h", :i)], ["n_h"]))
const DIFFN = Dict("stencil_loss" => ["u"], "matvec_loss" => ["a", "u"],
                   "mlp1d" => ["x", "y", "w1", "b1", "w2", "b2", "w3", "b3"])

# returns (kernel function, host arrays, integer arguments, shapes of the input files)
function make_case(case::String)
    if case == "stencil_loss" || case == "stencil_loss_big"
        n = case == "stencil_loss" ? 1000 : 40000
        host = Dict("loss" => zeros(1), "u" => 0.3 .* fgen(n, 0.013, 0.0, 0.1), "w" => zeros(n))
        return ("stencil_loss", host, Dict("i_n" => n), Dict("u" => [n]))
    elseif case == "matvec_loss"
        m, n = 30, 20
        host = Dict("loss" => zeros(1), "a" => reshape(0.3 .* fgen(m * n, 0.013, 0.0, 0.1), m, n),
                    "u" => 0.3 .* fgen(n, 0.026, 0.1, 0.1), "v" => zeros(m))
        return ("matvec_loss", host, Dict("i_m" => m, "i_n" => n), Dict("a" => [m, n], "u" => [n]))
    else
        nh = 64
        host = Dict("loss" => zeros(1), "x" => 0.3 .* fgen(1, 0.013, 0.0, 0.1), "y" => 0.3 .* fgen(1, 0.026, 0.1, 0.1),
                    "w1" => 0.3 .* fgen(nh, 0.039, 0.2, 0.1), "b1" => 0.3 .* fgen(nh, 0.052, 0.3, 0.1),
                    "w2" => 0.06 .* fgen(nh * nh, 0.065, 0.4, 0.1), "b2" => 0.3 .* fgen(nh, 0.078, 0.5, 0.1),
                    "w3" => 0.3 .* fgen(nh, 0.091, 0.6, 0.1), "b3" => 0.3 .* fgen(1, 0.104, 0.7, 0.1),
                    "h1" => zeros(nh), "h2" => zeros(nh), "o" => zeros(1))
        shapes = Dict(k => [length(host[k])] for k in ("x", "y", "w1", "b1", "w2", "b2", "w3", "b3"))
        return ("mlp1d", host, Dict("n_h" => nh), shapes)
    end
end

function load_variant(kfun::String, var::String)
    code = SRC[kfun][var]
    name = Symbol(kfun * "_" * var)
    info = Dict{String,Any}()
    try
        m = Module(name)
        Base.include_string(m, code, "gen_" * String(name))
        return (m, info)
    catch e
        info["first_error"] = first(sprint(showerror, e), 500)
    end
    # JACC.@init_backend may not exist in the installed JACC. Retry without that line and record it.
    code2 = join(filter(l -> !occursin("JACC.@init_backend", l), split(code, '\n')), '\n')
    m = Module(Symbol(String(name) * "_retry"))
    Base.include_string(m, code2, "gen_" * String(name) * "_retry")
    info["init_backend_line_removed"] = true
    return (m, info)
end

function run_variant(m, kfun, backend, T, wrap, host, ints)
    spec, stackargs = SPECS[kfun]
    args = Any[]
    arrs = Dict{String,Any}()
    for (nm, kind) in spec
        if kind == :i
            push!(args, ints[nm])
        else
            v = wrap(T.(host[nm]))
            s = wrap(nm == "loss" ? T[1] : zeros(T, size(host[nm])))
            push!(args, v)
            push!(args, s)
            arrs[nm] = (v, s)
        end
    end
    ini = getfield(m, Symbol("initstacks_$(kfun)_b_$(backend)"))
    st = Base.invokelatest(ini, [ints[n] for n in stackargs]...)
    stacks = st === nothing ? () : (st isa Tuple ? st : (st,))
    f = getfield(m, Symbol("$(kfun)_b_$(backend)"))
    Base.invokelatest(f, args..., stacks...)
    CUDA.synchronize()
    return (arrs, stacks)
end

function write_outputs(dir, case, var, kfun, arrs)
    for nm in DIFFN[kfun]
        open(joinpath(dir, "out_$(case)_$(var)_$(nm).bin"), "w") do io
            write(io, vec(Float64.(Array(arrs[nm][2]))))
        end
    end
    open(joinpath(dir, "out_$(case)_$(var)_loss.bin"), "w") do io
        write(io, vec(Float64.(Array(arrs["loss"][1]))))
    end
end

function run_py(dir, launch_text, args::Vector{String})
    sh = joinpath(dir, "launch_" * string(abs(hash(launch_text))) * ".sh")
    write(sh, launch_text)
    out = IOBuffer(); err = IOBuffer()
    p = run(pipeline(ignorestatus(`bash $sh $(joinpath(dir, "check_sp.py")) $args $dir`), stdout=out, stderr=err))
    so = String(take!(out)); se = String(take!(err))
    lines = filter(!isempty, strip.(split(so, '\n')))
    res = Dict{String,Any}("exit" => p.exitcode, "stderr_tail" => se[max(1, end - 600):end])
    try
        res["data"] = JSON3.read(lines[end], Dict{String,Any})
    catch
        res["data"] = Dict("error" => "could not parse output", "stdout_tail" => so[max(1, end - 600):end])
    end
    return res
end

function main()
    result = Dict{String,Any}("hostname" => gethostname())
    println(stderr, "NODE: ", gethostname()); flush(stderr)
    result["cuda"] = Dict("driver" => string(CUDA.driver_version()), "runtime" => string(CUDA.runtime_version()),
                          "device" => CUDA.name(CUDA.device()))
    dir = joinpath(get(ENV, "WORK", tempdir()), "bench", "spcheck-" * string(getpid()))
    mkpath(dir)
    result["dir"] = dir
    cases = ["stencil_loss", "stencil_loss_big", "matvec_loss", "mlp1d"]
    meta = Dict{String,Any}("kernels" => cases, "shapes" => Dict{String,Any}(), "ints" => Dict{String,Any}(),
                            "variants" => ["cuda_f64", "cuda_f32", "jacc_f64", "jacc_f32"])
    variants = [("cuda_f64", "cuda", Float64, CuArray), ("cuda_f32", "cuda", Float32, CuArray),
                ("jacc_f64", "jacc", Float64, JACC.array), ("jacc_f32", "jacc", Float32, JACC.array)]
    jinfo = Dict{String,Any}()
    for case in cases
        kfun, host, ints, shapes = make_case(case)
        meta["shapes"][case] = shapes
        meta["ints"][kfun] = ints
        for (nm, shp) in shapes
            open(joinpath(dir, "in_$(case)_$(nm).bin"), "w") do io
                write(io, vec(host[nm]))
            end
        end
        for (var, backend, T, wrap) in variants
            key = case * "/" * var
            try
                (m, info) = load_variant(kfun, var)
                (arrs, stacks) = run_variant(m, kfun, backend, T, wrap, host, ints)
                write_outputs(dir, case, var, kfun, arrs)
                first_name = DIFFN[kfun][1]
                info["status"] = "ok"
                info["array_type"] = string(typeof(arrs[first_name][1]))
                info["stack_eltypes"] = [string(eltype(s)) for s in stacks]
                info["loss_value"] = Float64(Array(arrs["loss"][1])[1])
                jinfo[key] = info
            catch e
                jinfo[key] = Dict("status" => "error", "error" => first(sprint(showerror, e), 600))
            end
        end
    end
    result["stade"] = jinfo
    open(joinpath(dir, "meta.json"), "w") do io
        write(io, JSON3.write(meta))
    end
    write(joinpath(dir, "check_sp.py"), PY_SCRIPT)
    result["torch"] = run_py(dir, TORCH_LAUNCH, ["torch"])
    result["jax64"] = run_py(dir, JAX_LAUNCH, ["jax64"])
    result["jax32"] = run_py(dir, JAX_LAUNCH, ["jax32"])
    return result
end

println(JSON3.write(main()))
