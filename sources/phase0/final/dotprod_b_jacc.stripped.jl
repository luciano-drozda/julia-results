using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_dotprod_b_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_dotprod_b_2!(__jacc_i, i_n, loss, u, v)
    i_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += u[i_x] * v[i_x]
    return nothing
end

function jacc_kernel_dotprod_b_3!(__jacc_i, i_n, lossb, u, ub, v, vb)
    i_x = i_n + (__jacc_i - 1) * -1
    __cse_0 = lossb[1]
    ub[i_x] = ub[i_x] + v[i_x] * __cse_0
    vb[i_x] = vb[i_x] + u[i_x] * __cse_0
    return nothing
end

function jacc_kernel_dotprod_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_dotprod_2!(__jacc_i, i_n, loss, u, v)
    i_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += u[i_x] * v[i_x]
    return nothing
end

function initstacks_dotprod_b_jacc()
    return nothing
end

function dotprod_b_jacc(loss, lossb, u, ub, v, vb, i_n)
    if div(i_n - 1, 1) + 1 < 32768
        if div(i_n - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_dotprod_b_2!(i_n, loss, u, v)
        end
    else
        if div(i_n - 1, 1) + 1 > 0
            __jgen_redval_1 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_x, u, v)->u[i_x] * v[i_x]))(u, v))
            JACC.@parallel_for range = 1 jacc_kernel_dotprod_b_1!(loss, __jgen_redval_1)
        end
    end
    if div(1 - i_n, -1) + 1 > 0
        JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_dotprod_b_3!(i_n, lossb, u, ub, v, vb)
    end
    return nothing
end

function dotprod_jacc(loss, u, v, i_n)
    if div(i_n - 1, 1) + 1 < 32768
        if div(i_n - 1, 1) + 1 > 0
            JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_dotprod_2!(i_n, loss, u, v)
        end
    else
        if div(i_n - 1, 1) + 1 > 0
            __jgen_redval_1 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_x, u, v)->u[i_x] * v[i_x]))(u, v))
            JACC.@parallel_for range = 1 jacc_kernel_dotprod_1!(loss, __jgen_redval_1)
        end
    end
    return nothing
end
