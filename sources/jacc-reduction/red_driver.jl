# Reduction kernels with lower bound / stride / reverse direction, forced onto the reduce path (reduction_threshold = 0).
using CUDA, JSON3
import JACC
const SRC = Dict{String,String}()
SRC["red_lo2_cuda"] = raw"""using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_red_lo2_1!(i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    CUDA.@atomic loss[1] += u[i_x] ^ 2
    return nothing
end

function red_lo2_cuda(u, loss, i_n)
    nthread_per_block = 256
    if div(i_n - 2, 1) + 1 < 0
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_n - 2, 1) + 1, nthread_per_block)) cuda_kernel_red_lo2_1!(i_n, loss, u)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(u, 2:i_n))), abs2, view(u, 2:i_n))
            end
    end
    return nothing
end
"""
SRC["red_lo2_jacc"] = raw"""using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_red_lo2_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_red_lo2_2!(__jacc_i, i_n, loss, u)
    i_x = 2 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += u[i_x] ^ 2
    return nothing
end

function red_lo2_jacc(u, loss, i_n)
    if div(i_n - 2, 1) + 1 < 0
        if div(i_n - 2, 1) + 1 > 0
            JACC.@parallel_for range = div(i_n - 2, 1) + 1 jacc_kernel_red_lo2_2!(i_n, loss, u)
        end
    else
        __jgen_redval_1 = JACC.@parallel_reduce(range = div(i_n - 2, 1) + 1, (((i_x, u)->u[i_x] ^ 2))(u))
        JACC.@parallel_for range = 1 jacc_kernel_red_lo2_1!(loss, __jgen_redval_1)
    end
    return nothing
end
"""
SRC["red_step2_cuda"] = raw"""using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_red_step2_1!(i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > max(0, div((i_n - 1) + 2, 2))
        return nothing
    end
    i_x = 1 + (__tid - 1) * 2
    CUDA.@atomic loss[1] += u[i_x] ^ 2
    return nothing
end

function red_step2_cuda(u, loss, i_n)
    nthread_per_block = 256
    if max(0, div((i_n - 1) + 2, 2)) < 0
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(max(0, div((i_n - 1) + 2, 2)), nthread_per_block)) cuda_kernel_red_step2_1!(i_n, loss, u)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(u, 1:2:i_n))), abs2, view(u, 1:2:i_n))
            end
    end
    return nothing
end
"""
SRC["red_step2_jacc"] = raw"""using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_red_step2_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_red_step2_2!(__jacc_i, i_n, loss, u)
    i_x = 1 + (__jacc_i - 1) * 2
    Atomix.@atomic loss[1] += u[i_x] ^ 2
    return nothing
end

function red_step2_jacc(u, loss, i_n)
    if max(0, div((i_n - 1) + 2, 2)) < 0
        if max(0, div((i_n - 1) + 2, 2)) > 0
            JACC.@parallel_for range = max(0, div((i_n - 1) + 2, 2)) jacc_kernel_red_step2_2!(i_n, loss, u)
        end
    else
        __jgen_redval_1 = JACC.@parallel_reduce(range = max(0, div((i_n - 1) + 2, 2)), (((i_x, u)->u[i_x] ^ 2))(u))
        JACC.@parallel_for range = 1 jacc_kernel_red_step2_1!(loss, __jgen_redval_1)
    end
    return nothing
end
"""
SRC["red_rev3_cuda"] = raw"""using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_red_rev3_1!(i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(3 - i_n, -1) + 1
        return nothing
    end
    i_x = i_n + (__tid - 1) * -1
    CUDA.@atomic loss[1] += u[i_x] ^ 2
    return nothing
end

function red_rev3_cuda(u, loss, i_n)
    nthread_per_block = 256
    if div(3 - i_n, -1) + 1 < 0
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(3 - i_n, -1) + 1, nthread_per_block)) cuda_kernel_red_rev3_1!(i_n, loss, u)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(init = zero(eltype(view(u, i_n:-1:3))), abs2, view(u, i_n:-1:3))
            end
    end
    return nothing
end
"""
SRC["red_rev3_jacc"] = raw"""using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_red_rev3_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_red_rev3_2!(__jacc_i, i_n, loss, u)
    i_x = i_n + (__jacc_i - 1) * -1
    Atomix.@atomic loss[1] += u[i_x] ^ 2
    return nothing
end

function red_rev3_jacc(u, loss, i_n)
    if div(3 - i_n, -1) + 1 < 0
        if div(3 - i_n, -1) + 1 > 0
            JACC.@parallel_for range = div(3 - i_n, -1) + 1 jacc_kernel_red_rev3_2!(i_n, loss, u)
        end
    else
        __jgen_redval_1 = JACC.@parallel_reduce(range = div(3 - i_n, -1) + 1, (((i_x, u)->u[i_x] ^ 2))(u))
        JACC.@parallel_for range = 1 jacc_kernel_red_rev3_1!(loss, __jgen_redval_1)
    end
    return nothing
end
"""
SRC["red_lo3s3_cuda"] = raw"""using CUDA
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
"""
SRC["red_lo3s3_jacc"] = raw"""using CUDA
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_red_lo3s3_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_red_lo3s3_2!(__jacc_i, i_n, loss, u)
    i_x = 3 + (__jacc_i - 1) * 3
    Atomix.@atomic loss[1] += u[i_x] ^ 2
    return nothing
end

function red_lo3s3_jacc(u, loss, i_n)
    if max(0, div((i_n - 3) + 3, 3)) < 0
        if max(0, div((i_n - 3) + 3, 3)) > 0
            JACC.@parallel_for range = max(0, div((i_n - 3) + 3, 3)) jacc_kernel_red_lo3s3_2!(i_n, loss, u)
        end
    else
        __jgen_redval_1 = JACC.@parallel_reduce(range = max(0, div((i_n - 3) + 3, 3)), (((i_x, u)->u[i_x] ^ 2))(u))
        JACC.@parallel_for range = 1 jacc_kernel_red_lo3s3_1!(loss, __jgen_redval_1)
    end
    return nothing
end
"""

cpu_range(k, n) = k == "red_lo2" ? (2:n) : k == "red_step2" ? (1:2:n) : k == "red_rev3" ? (n:-1:3) : (3:3:n)
# trip count zero
zero_n(k) = k == "red_lo2" ? 1 : k == "red_step2" ? 0 : k == "red_rev3" ? 2 : 2

function run_one(m, k, be, wrap, u, n)
    loss = wrap(zeros(1))
    f = getfield(m, Symbol(k * "_" * be))
    Base.invokelatest(f, wrap(u), loss, n)
    CUDA.synchronize()
    return Array(loss)[1]
end

function main()
    res = Dict{String,Any}("hostname" => gethostname())
    println(stderr, "NODE: ", gethostname()); flush(stderr)
    u = [0.5 + sin(0.1 * i) for i in 1:2000]
    rows = Any[]
    for k in ("red_lo2", "red_step2", "red_rev3", "red_lo3s3")
        for be in ("cuda", "jacc")
            wrap = be == "cuda" ? CuArray : JACC.array
            m = Module(Symbol("M_" * k * "_" * be)); Base.include_string(m, SRC[k * "_" * be], "gen_" * k * be)
            for n in (100, 1000, zero_n(k))
                row = Dict{String,Any}("kernel" => k, "backend" => be, "n" => n)
                expected = sum(u[i]^2 for i in cpu_range(k, n); init = 0.0)
                row["expected"] = expected; row["trip_count"] = length(cpu_range(k, n))
                try
                    got = run_one(m, k, be, wrap, u, n)
                    row["got"] = got
                    row["rel_err"] = abs(got - expected) / max(abs(expected), 1e-300)
                    row["abs_err"] = abs(got - expected)
                catch e
                    row["exception"] = first(replace(sprint(showerror, e), r"\s+" => " "), 300)
                end
                push!(rows, row)
            end
        end
    end
    res["rows"] = rows
    return res
end
println(JSON3.write(main()))
