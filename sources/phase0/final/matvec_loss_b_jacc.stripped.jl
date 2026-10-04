using CUDA
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
        if div(i_m - 1, 1) + 1 > 0
            __jgen_redval_2 = JACC.@parallel_reduce(range = div(i_m - 1, 1) + 1, (((i_i2, v)->v[i_i2] ^ 2))(v))
            JACC.@parallel_for range = 1 jacc_kernel_matvec_loss_b_2!(loss, __jgen_redval_2)
        end
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
        if div(i_m - 1, 1) + 1 > 0
            __jgen_redval_2 = JACC.@parallel_reduce(range = div(i_m - 1, 1) + 1, (((i_i2, v)->v[i_i2] ^ 2))(v))
            JACC.@parallel_for range = 1 jacc_kernel_matvec_loss_2!(loss, __jgen_redval_2)
        end
    end
    return nothing
end
