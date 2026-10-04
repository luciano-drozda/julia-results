# dotprod and matvec_loss: CUDA and JACC, primal and adjoint, sizes below and above the reduce threshold (32768).
using CUDA, JSON3
import JACC
const SRC = Dict{String,String}()
SRC["dotprod_cuda"] = raw"""using CUDA
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
    if div(i_n - 1, 1) + 1 < 32768
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
    if div(i_n - 1, 1) + 1 < 32768
        @cuda threads = nthread_per_block blocks = max(1, Base.cld(div(i_n - 1, 1) + 1, nthread_per_block)) cuda_kernel_dotprod_1!(i_n, loss, u, v)
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + mapreduce(init = zero(eltype(view(u, 1:i_n))), ((__mr_1, __mr_2)->__mr_1 * __mr_2), +, view(u, 1:i_n), view(v, 1:i_n))
            end
    end
    return nothing
end
"""
SRC["dotprod_jacc"] = raw"""using CUDA
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
"""
SRC["matvec_loss_cuda"] = raw"""using CUDA
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
SRC["matvec_loss_jacc"] = raw"""using CUDA
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
"""

relerr(a, b) = maximum(abs.(Float64.(a) .- b)) / max(maximum(abs.(b)), 1e-300)

function dot_case(mods, be, wrap, n)
    u = [0.5 + sin(0.1 * i) for i in 1:n]; v = [cos(0.07 * i) for i in 1:n]
    lref = sum(u .* v); r = Dict{String,Any}()
    m = mods[be]
    loss = wrap(zeros(1)); Base.invokelatest(getfield(m, Symbol("dotprod_$(be)")), loss, wrap(u), wrap(v), n); CUDA.synchronize()
    r["primal_loss_err"] = abs(Array(loss)[1] - lref) / abs(lref)
    loss = wrap(zeros(1)); ub = wrap(zeros(n)); vb = wrap(zeros(n))
    Base.invokelatest(getfield(m, Symbol("dotprod_b_$(be)")), loss, wrap([1.0]), wrap(u), ub, wrap(v), vb, n); CUDA.synchronize()
    r["adjoint_loss_err"] = abs(Array(loss)[1] - lref) / abs(lref)
    r["adjoint_ub_err"] = relerr(Array(ub), v); r["adjoint_vb_err"] = relerr(Array(vb), u)
    return r
end

function mv_case(mods, be, wrap, mm, nn)
    A = [sin(0.01 * i + 0.3 * j) for i in 1:mm, j in 1:nn]; u = [0.4 + cos(0.2 * j) for j in 1:nn]
    v = A * u; lref = sum(v .^ 2); abr = 2 .* v * u'; ubr = 2 .* (A' * v); r = Dict{String,Any}()
    m = mods[be]
    loss = wrap(zeros(1)); Base.invokelatest(getfield(m, Symbol("matvec_loss_$(be)")), loss, wrap(A), wrap(u), wrap(zeros(mm)), mm, nn); CUDA.synchronize()
    r["primal_loss_err"] = abs(Array(loss)[1] - lref) / abs(lref)
    loss = wrap(zeros(1)); ab = wrap(zeros(mm, nn)); ubd = wrap(zeros(nn))
    Base.invokelatest(getfield(m, Symbol("matvec_loss_b_$(be)")), loss, wrap([1.0]), wrap(A), ab, wrap(u), ubd, wrap(zeros(mm)), wrap(zeros(mm)), mm, nn); CUDA.synchronize()
    r["adjoint_loss_err"] = abs(Array(loss)[1] - lref) / abs(lref)
    r["adjoint_ab_err"] = relerr(Array(ab), abr); r["adjoint_ub_err"] = relerr(Array(ubd), ubr)
    return r
end

function main()
    res = Dict{String,Any}("hostname" => gethostname())
    println(stderr, "NODE: ", gethostname()); flush(stderr)
    rows = Any[]
    for (kern, caseset) in (("dotprod", [(1000,), (40000,), (1000000,)]), ("matvec_loss", [(300, 20), (40000, 4), (200000, 3)]))
        for be in ("cuda", "jacc")
            wrap = be == "cuda" ? CuArray : JACC.array
            mods = Dict{String,Any}()
            m = Module(Symbol("M_" * kern * "_" * be)); Base.include_string(m, SRC[kern * "_" * be], "gen_" * kern * be); mods[be] = m
            for cs in caseset
                row = Dict{String,Any}("kernel" => kern, "backend" => be, "size" => collect(cs))
                try
                    row["result"] = kern == "dotprod" ? dot_case(mods, be, wrap, cs[1]) : mv_case(mods, be, wrap, cs[1], cs[2])
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
