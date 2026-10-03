# Diagnostic: is the stencil_loss adjoint wrong because of a data race in the generated CUDA kernel?
using CUDA, JSON3
const SRC_ORIG = raw"""using CUDA
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
const PATCHES = [
    "    ub[i_x - 1] = ub[i_x - 1] + __oldb_0" => "    CUDA.@atomic ub[i_x - 1] += __oldb_0",
    "    ub[i_x] = ub[i_x] + 2.0 * -__oldb_0" => "    CUDA.@atomic ub[i_x] += 2.0 * -__oldb_0",
    "    ub[i_x + 1] = ub[i_x + 1] + __oldb_0" => "    CUDA.@atomic ub[i_x + 1] += __oldb_0"]

function cpu_ref(u)
    n = length(u); g = zeros(n); loss = 0.0
    for i in 2:n-1
        w = u[i-1] - 2 * u[i] + u[i+1]
        loss += w^2; g[i-1] += 2w; g[i] -= 4w; g[i+1] += 2w
    end
    return loss, g
end

function run_one(m, u)
    n = length(u)
    loss = CuArray(zeros(1)); lossb = CuArray([1.0]); ud = CuArray(u); ub = CuArray(zeros(n)); w = CuArray(zeros(n)); wb = CuArray(zeros(n))
    Base.invokelatest(m.stencil_loss_b_cuda, loss, lossb, ud, ub, w, wb, n)
    CUDA.synchronize()
    return Array(loss)[1], Array(ub)
end

function main()
    res = Dict{String,Any}("hostname" => gethostname())
    println(stderr, "NODE: ", gethostname()); flush(stderr)
    m0 = Module(:Orig); Base.include_string(m0, SRC_ORIG, "orig")
    patched = SRC_ORIG
    for p in PATCHES
        patched = replace(patched, p)
    end
    res["patch_applied"] = patched != SRC_ORIG
    res["atomic_lines_in_patched"] = count("CUDA.@atomic ub[", patched) - count("CUDA.@atomic ub[", SRC_ORIG)
    res["atomic_lines_in_original"] = count("CUDA.@atomic ub[", SRC_ORIG)
    if !res["patch_applied"]
        return res
    end
    m1 = Module(:Patched); Base.include_string(m1, patched, "patched")
    sizes = [4, 32, 100, 256, 512, 1000, 40000]
    rows = Any[]
    for n in sizes
        u = 0.3 .* [sin(0.013 * i) + 0.1 * cos(0.7 * i) for i in 1:n]
        lref, gref = cpu_ref(u)
        sc = max(maximum(abs.(gref)), 1e-300)
        e_orig = Float64[]; e_patch = Float64[]
        for rep in 1:8
            _, g0 = run_one(m0, u); push!(e_orig, maximum(abs.(g0 .- gref)) / sc)
        end
        for rep in 1:8
            _, g1 = run_one(m1, u); push!(e_patch, maximum(abs.(g1 .- gref)) / sc)
        end
        push!(rows, Dict("n" => n, "orig_errors" => e_orig, "patched_errors" => e_patch))
    end
    res["rows"] = rows
    return res
end
println(JSON3.write(main()))
