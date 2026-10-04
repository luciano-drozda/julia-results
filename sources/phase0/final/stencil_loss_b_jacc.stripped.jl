using CUDA
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
    Atomix.@atomic ub[i_x - 1] += __oldb_0
    Atomix.@atomic ub[i_x] += 2.0 * -__oldb_0
    Atomix.@atomic ub[i_x + 1] += __oldb_0
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
        if div((i_n - 1) - 2, 1) + 1 > 0
            __jgen_redval_2 = JACC.@parallel_reduce(range = div((i_n - 1) - 2, 1) + 1, (((__jgen_k, w)->w[2 + (__jgen_k - 1)] ^ 2))(w))
            JACC.@parallel_for range = 1 jacc_kernel_stencil_loss_b_2!(loss, __jgen_redval_2)
        end
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
        if div((i_n - 1) - 2, 1) + 1 > 0
            __jgen_redval_2 = JACC.@parallel_reduce(range = div((i_n - 1) - 2, 1) + 1, (((__jgen_k, w)->w[2 + (__jgen_k - 1)] ^ 2))(w))
            JACC.@parallel_for range = 1 jacc_kernel_stencil_loss_2!(loss, __jgen_redval_2)
        end
    end
    return nothing
end
