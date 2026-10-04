using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_advection_b_1!(__jacc_i, du, du_stack, i_, i_nnode, u)
    i_x = 2 + (__jacc_i - 1)
    __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
    du_stack[__idx_du_stack_0] = du[i_x]
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function jacc_kernel_advection_b_2!(__jacc_i, c, dt, du, dx, i_nnode, u)
    i_x = 2 + (__jacc_i - 1)
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function jacc_kernel_advection_b_3!(__jacc_i, c, cb, dt, dtb, du, dub, dx, dxb, i_nnode, ub)
    i_x = i_nnode + (__jacc_i - 1) * -1
    __cse_0 = du[i_x]
    __cse_1 = -(ub[i_x])
    __cse_2 = (1.0 / dx) * __cse_1
    Atomix.@atomic cb[1] += (dt * __cse_0) * __cse_2
    Atomix.@atomic dtb[1] += (c * __cse_0) * __cse_2
    dub[i_x] = dub[i_x] + (c * dt) * __cse_2
    Atomix.@atomic dxb[1] += -((c * dt * __cse_0) / dx ^ 2) * __cse_1
    return nothing
end

function jacc_kernel_advection_b_4!(__jacc_i, du, du_stack, dub, i_, i_nnode, ub)
    i_x = i_nnode + (__jacc_i - 1) * -1
    __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
    du[i_x] = du_stack[__idx_du_stack_0]
    __oldb_2 = dub[i_x]
    dub[i_x] = 0.0
    Atomix.@atomic ub[i_x] += __oldb_2
    Atomix.@atomic ub[i_x - 1] += -__oldb_2
    return nothing
end

function jacc_kernel_advection_1!(__jacc_i, du, i_nnode, u)
    i_x = 2 + (__jacc_i - 1)
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function jacc_kernel_advection_2!(__jacc_i, c, dt, du, dx, i_nnode, u)
    i_x = 2 + (__jacc_i - 1)
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function initstacks_advection_b_jacc(i_nnode, i_nstep)
    du_stack = JACC.zeros(Float64, max(0, div(i_nstep - 1, 1) + 1) * max(0, div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_b_jacc(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    cb = JACC.array([cb])
    dtb = JACC.array([dtb])
    dxb = JACC.array([dxb])
    for i_ = 1:i_nstep
        if div(i_nnode - 2, 1) + 1 > 0
            JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_b_1!(du, du_stack, i_, i_nnode, u)
        end
        if div(i_nnode - 2, 1) + 1 > 0
            JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_b_2!(c, dt, du, dx, i_nnode, u)
        end
    end
    for i_ = i_nstep:-1:1
        if div(2 - i_nnode, -1) + 1 > 0
            JACC.@parallel_for range = div(2 - i_nnode, -1) + 1 jacc_kernel_advection_b_3!(c, cb, dt, dtb, du, dub, dx, dxb, i_nnode, ub)
        end
        if div(2 - i_nnode, -1) + 1 > 0
            JACC.@parallel_for range = div(2 - i_nnode, -1) + 1 jacc_kernel_advection_b_4!(du, du_stack, dub, i_, i_nnode, ub)
        end
    end
    cb = (JACC.to_host(cb))[1]
    dtb = (JACC.to_host(dtb))[1]
    dxb = (JACC.to_host(dxb))[1]
    return (cb, dxb, dtb)
end

function advection_jacc(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_ = 1:i_nstep
        if div(i_nnode - 2, 1) + 1 > 0
            JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_1!(du, i_nnode, u)
        end
        if div(i_nnode - 2, 1) + 1 > 0
            JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
        end
    end
    return nothing
end
