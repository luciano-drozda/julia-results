# advection (run 2: du is compared with its RESTORED initial value, as STADE's CPU adjoint also leaves it): is the generated GPU primal (and adjoint) race-free? Compares against hand-written CPU references.
using CUDA, JSON3
import JACC

const SRC = Dict{String,String}()
SRC["cuda_f64"] = raw"""using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_advection_b_1!(du, du_stack, i_, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
    du_stack[__idx_du_stack_0] = du[i_x]
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function cuda_kernel_advection_b_2!(c, dt, du, dx, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function cuda_kernel_advection_b_3!(c, cb, dt, dtb, du, dub, dx, dxb, i_nnode, ub)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(2 - i_nnode, -1) + 1
        return nothing
    end
    i_x = i_nnode + (__tid - 1) * -1
    __cse_0 = du[i_x]
    __cse_1 = -(ub[i_x])
    __cse_2 = (1.0 / dx) * __cse_1
    CUDA.@atomic cb[1] += (dt * __cse_0) * __cse_2
    CUDA.@atomic dtb[1] += (c * __cse_0) * __cse_2
    dub[i_x] = dub[i_x] + (c * dt) * __cse_2
    CUDA.@atomic dxb[1] += -((c * dt * __cse_0) / dx ^ 2) * __cse_1
    return nothing
end

function cuda_kernel_advection_b_4!(du, du_stack, dub, i_, i_nnode, ub)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(2 - i_nnode, -1) + 1
        return nothing
    end
    i_x = i_nnode + (__tid - 1) * -1
    __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
    du[i_x] = du_stack[__idx_du_stack_0]
    __oldb_2 = dub[i_x]
    dub[i_x] = 0.0
    CUDA.@atomic ub[i_x] += __oldb_2
    CUDA.@atomic ub[i_x - 1] += -__oldb_2
    return nothing
end

function cuda_kernel_advection_1!(du, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function cuda_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function initstacks_advection_b_cuda(i_nnode, i_nstep)
    du_stack = CuArray{Float64}(undef, max(0, div(i_nstep - 1, 1) + 1) * max(0, div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_b_cuda(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    cb = CuArray([cb])
    dtb = CuArray([dtb])
    dxb = CuArray([dxb])
    nthread_per_block = 256
    for i_ = 1:i_nstep
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_nnode - 2, 1) + 1, nthread_per_block)) cuda_kernel_advection_b_1!(du, du_stack, i_, i_nnode, u)
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_nnode - 2, 1) + 1, nthread_per_block)) cuda_kernel_advection_b_2!(c, dt, du, dx, i_nnode, u)
    end
    for i_ = i_nstep:-1:1
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(2 - i_nnode, -1) + 1, nthread_per_block)) cuda_kernel_advection_b_3!(c, cb, dt, dtb, du, dub, dx, dxb, i_nnode, ub)
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(2 - i_nnode, -1) + 1, nthread_per_block)) cuda_kernel_advection_b_4!(du, du_stack, dub, i_, i_nnode, ub)
    end
    cb = (Array(cb))[1]
    dtb = (Array(dtb))[1]
    dxb = (Array(dxb))[1]
    return (cb, dxb, dtb)
end

function advection_cuda(u, du, c, dx, dt, i_nstep, i_nnode)
    nthread_per_block = 256
    for i_ = 1:i_nstep
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_nnode - 2, 1) + 1, nthread_per_block)) cuda_kernel_advection_1!(du, i_nnode, u)
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_nnode - 2, 1) + 1, nthread_per_block)) cuda_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
"""
SRC["cuda_f32"] = raw"""using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_advection_b_1!(du, du_stack, i_, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
    du_stack[__idx_du_stack_0] = du[i_x]
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function cuda_kernel_advection_b_2!(c, dt, du, dx, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function cuda_kernel_advection_b_3!(c, cb, dt, dtb, du, dub, dx, dxb, i_nnode, ub)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(2 - i_nnode, -1) + 1
        return nothing
    end
    i_x = i_nnode + (__tid - 1) * -1
    __cse_0 = du[i_x]
    __cse_1 = -(ub[i_x])
    __cse_2 = (1.0f0 / dx) * __cse_1
    CUDA.@atomic cb[1] += (dt * __cse_0) * __cse_2
    CUDA.@atomic dtb[1] += (c * __cse_0) * __cse_2
    dub[i_x] = dub[i_x] + (c * dt) * __cse_2
    CUDA.@atomic dxb[1] += -((c * dt * __cse_0) / dx ^ 2) * __cse_1
    return nothing
end

function cuda_kernel_advection_b_4!(du, du_stack, dub, i_, i_nnode, ub)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(2 - i_nnode, -1) + 1
        return nothing
    end
    i_x = i_nnode + (__tid - 1) * -1
    __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
    du[i_x] = du_stack[__idx_du_stack_0]
    __oldb_2 = dub[i_x]
    dub[i_x] = 0.0f0
    CUDA.@atomic ub[i_x] += __oldb_2
    CUDA.@atomic ub[i_x - 1] += -__oldb_2
    return nothing
end

function cuda_kernel_advection_1!(du, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function cuda_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function initstacks_advection_b_cuda(i_nnode, i_nstep)
    du_stack = CuArray{Float32}(undef, max(0, div(i_nstep - 1, 1) + 1) * max(0, div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_b_cuda(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    cb = CuArray([cb])
    dtb = CuArray([dtb])
    dxb = CuArray([dxb])
    nthread_per_block = 256
    for i_ = 1:i_nstep
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_nnode - 2, 1) + 1, nthread_per_block)) cuda_kernel_advection_b_1!(du, du_stack, i_, i_nnode, u)
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_nnode - 2, 1) + 1, nthread_per_block)) cuda_kernel_advection_b_2!(c, dt, du, dx, i_nnode, u)
    end
    for i_ = i_nstep:-1:1
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(2 - i_nnode, -1) + 1, nthread_per_block)) cuda_kernel_advection_b_3!(c, cb, dt, dtb, du, dub, dx, dxb, i_nnode, ub)
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(2 - i_nnode, -1) + 1, nthread_per_block)) cuda_kernel_advection_b_4!(du, du_stack, dub, i_, i_nnode, ub)
    end
    cb = (Array(cb))[1]
    dtb = (Array(dtb))[1]
    dxb = (Array(dxb))[1]
    return (cb, dxb, dtb)
end

function advection_cuda(u, du, c, dx, dt, i_nstep, i_nnode)
    nthread_per_block = 256
    for i_ = 1:i_nstep
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_nnode - 2, 1) + 1, nthread_per_block)) cuda_kernel_advection_1!(du, i_nnode, u)
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_nnode - 2, 1) + 1, nthread_per_block)) cuda_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
"""
SRC["jacc_f64"] = raw"""using CUDA
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
"""
SRC["jacc_f32"] = raw"""using CUDA
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
    __cse_2 = (1.0f0 / dx) * __cse_1
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
    dub[i_x] = 0.0f0
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
    du_stack = JACC.zeros(Float32, max(0, div(i_nstep - 1, 1) + 1) * max(0, div(i_nnode - 2, 1) + 1))
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
"""

# Single-line patches: make the two plain shadow updates atomic. Each must match exactly once.
const PATCHES = Dict(
    "cuda" => ["    ub[i_x] = ub[i_x] + __oldb_2" => "    CUDA.@atomic ub[i_x] += __oldb_2",
               "    ub[i_x - 1] = ub[i_x - 1] + -__oldb_2" => "    CUDA.@atomic ub[i_x - 1] += -__oldb_2"],
    "jacc" => ["    ub[i_x] = ub[i_x] + __oldb_2" => "    Atomix.@atomic ub[i_x] += __oldb_2",
               "    ub[i_x - 1] = ub[i_x - 1] + -__oldb_2" => "    Atomix.@atomic ub[i_x - 1] += -__oldb_2"])

ufun(n) = [sin(0.013 * i) + 0.1 * cos(0.7 * i) for i in 1:n]
dufun(n) = [0.1 * sin(0.05 * i) for i in 1:n]
ubfun(n) = [0.5 * cos(0.011 * i) for i in 1:n]
dubfun(n) = [0.3 * sin(0.017 * i) for i in 1:n]

# sequential Float64 reference of the primal
function cpu_primal(u0, du0, c, dx, dt, nstep)
    n = length(u0); u = copy(u0); du = copy(du0)
    for s in 1:nstep
        for i in 2:n
            du[i] = u[i] - u[i-1]
        end
        for i in 2:n
            u[i] = u[i] - (c * dt * du[i]) / dx
        end
    end
    return (u, du)
end

# sequential Float64 reference of the adjoint (hand-derived reverse sweep with the history of u)
function cpu_adjoint(u0, du0, ubs, dubs, c, dx, dt, nstep)
    n = length(u0); k = c * dt / dx
    hist = Vector{Vector{Float64}}(); u = copy(u0); du = copy(du0)
    for s in 1:nstep
        push!(hist, copy(u))
        for i in 2:n
            du[i] = u[i] - u[i-1]
        end
        for i in 2:n
            u[i] = u[i] - (c * dt * du[i]) / dx
        end
    end
    ub = copy(ubs); dub = copy(dubs); kb = 0.0
    for s in nstep:-1:1
        us = hist[s]
        for i in 2:n
            dus = us[i] - us[i-1]
            dub[i] += -k * ub[i]
            kb += -dus * ub[i]
        end
        for i in n:-1:2
            d = dub[i]; dub[i] = 0.0
            ub[i] += d; ub[i-1] -= d
        end
    end
    return (u, du, ub, dub, kb * dt / dx, -kb * c * dt / dx^2, kb * c / dx)   # u, du, ub, dub, cb, dxb, dtb
end

relerr(a, b) = maximum(abs.(Float64.(a) .- b)) / max(maximum(abs.(b)), 1e-300)
srelerr(a, b) = abs(Float64(a) - b) / max(abs(b), 1e-300)

function load_module(code::String, name::String)
    m = Module(Symbol(name))
    Base.include_string(m, code, "gen_" * name)
    return m
end

function main()
    res = Dict{String,Any}("hostname" => gethostname())
    println(stderr, "NODE: ", gethostname()); flush(stderr)
    c0, dx0, dt0 = 0.4, 1.0, 0.1
    variants = Any[]
    for (key, be, T, wrap) in (("cuda_f64", "cuda", Float64, CuArray), ("cuda_f32", "cuda", Float32, CuArray),
                               ("jacc_f64", "jacc", Float64, JACC.array), ("jacc_f32", "jacc", Float32, JACC.array))
        push!(variants, (key, be, T, wrap, SRC[key], false))
    end
    mods = Dict{String,Any}()
    status = Dict{String,Any}()
    for (key, be, T, wrap, code, isp) in variants
        try
            mods[key] = load_module(code, key)
            status[key] = Dict("loaded" => true, "atomic_ub_lines" => count("@atomic ub[", code))
        catch e
            status[key] = Dict("loaded" => false, "error" => first(sprint(showerror, e), 500))
        end
    end
    res["variants"] = status
    cases = [(4, 5), (33, 5), (100, 20), (256, 20), (1000, 50), (40000, 50), (1000000, 5)]
    out = Dict{String,Any}()
    for (n, nstep) in cases
        ck = "n$(n)_s$(nstep)"
        u0 = ufun(n); du0 = dufun(n); ubs = ubfun(n); dubs = dubfun(n)
        (pu, pdu) = cpu_primal(u0, du0, c0, dx0, dt0, nstep)
        (au, adu, aub, adub, acb, adxb, adtb) = cpu_adjoint(u0, du0, ubs, dubs, c0, dx0, dt0, nstep)
        out[ck] = Dict{String,Any}()
        for (key, be, T, wrap, code, isp) in variants
            haskey(mods, key) || continue
            m = mods[key]
            rec = Dict{String,Any}()
            try
                prim = getfield(m, Symbol("advection_$(be)"))
                err_pu = Float64[]; err_pdu = Float64[]
                for rep in 1:5
                    u = wrap(T.(u0)); du = wrap(T.(du0))
                    Base.invokelatest(prim, u, du, T(c0), T(dx0), T(dt0), nstep, n)
                    CUDA.synchronize()
                    push!(err_pu, relerr(Array(u), pu)); push!(err_pdu, relerr(Array(du), pdu))
                end
                rec["primal_err_u"] = err_pu; rec["primal_err_du_max"] = maximum(err_pdu)
            catch e
                rec["primal_error"] = first(sprint(showerror, e), 400)
            end
            try
                adj = getfield(m, Symbol("advection_b_$(be)"))
                ini = getfield(m, Symbol("initstacks_advection_b_$(be)"))
                err_ub = Float64[]; e_u = 0.0; e_du = 0.0; e_dub = 0.0; err_sc = 0.0; stype = ""
                for rep in 1:5
                    u = wrap(T.(u0)); du = wrap(T.(du0)); ub = wrap(T.(ubs)); dub = wrap(T.(dubs))
                    st = Base.invokelatest(ini, n, nstep); stype = string(eltype(st))
                    (cb, dxb, dtb) = Base.invokelatest(adj, u, ub, du, dub, T(c0), zero(T), T(dx0), zero(T), T(dt0), zero(T), nstep, n, st)
                    CUDA.synchronize()
                    push!(err_ub, relerr(Array(ub), aub))
                    e_u = max(e_u, relerr(Array(u), au))
                    e_du = max(e_du, relerr(Array(du), du0))
                    e_dub = max(e_dub, relerr(Array(dub), adub))
                    err_sc = max(err_sc, srelerr(cb, acb), srelerr(dxb, adxb), srelerr(dtb, adtb))
                end
                rec["adjoint_err_ub"] = err_ub; rec["adjoint_err_u_final_max"] = e_u; rec["adjoint_err_du_restored_max"] = e_du; rec["adjoint_err_dub_max"] = e_dub
                rec["adjoint_err_scalars_max"] = err_sc; rec["tape_eltype"] = stype
            catch e
                rec["adjoint_error"] = first(sprint(showerror, e), 400)
            end
            out[ck][key] = rec
        end
    end
    res["cases"] = out
    return res
end
println(JSON3.write(main()))
