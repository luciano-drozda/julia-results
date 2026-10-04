# STADE 0.4.3: data race in the generated GPU adjoints of `stencil_loss` and `advection`

> **Status 2026-10-04: FIXED and verified on the V100** (STADE zip SHA-256 `d60e495a...`). See "Verification of the fix" and the open item "JACC reduction omits one term" at the end of this note.

Found on 2026-10-03 during a single-precision health check. Tesla V100-PCIE-16GB, CUDA.jl, JACC 1.3.1, Julia 1.11.9 (code generated with Julia 1.10.11).
Raw data: branch `bench-raw` of `luciano-drozda/julia-results`, folder `raw/healthchecks/` (jobs `healthcheck-sp-v100-*`, `diag-stencil-race2-*`) and `sources/healthcheck-singleprecision/`.

## Symptom
`stencil_loss_b_cuda` and `stencil_loss_b_jacc` return the correct loss and a wrong gradient `ub`.
The error is the same in Float64 and Float32, so it is not a precision effect.

| n | Original generated kernel (8 runs, max relative error) | With atomic updates |
|---|---|---|
| 4, 32, 100 | 0 to 1e-16 in all runs | same |
| 256 | 9e-17, 9e-2, 7e-1, 4e-1, ... (different answers for the same input) | 9e-17 in all runs |
| 512 | 0.7 to 1.0 | 9e-17 |
| 1000 | 1.0 to 2.0 | 1e-16 |
| 40000 | 2.0 | 3e-16 |

## Cause (confirmed by the patch below)
`cuda_kernel_stencil_loss_b_4!` runs one thread per iteration and contains:

```julia
ub[i_x - 1] = ub[i_x - 1] + __oldb_0
ub[i_x] = ub[i_x] + 2.0 * -__oldb_0
ub[i_x + 1] = ub[i_x + 1] + __oldb_0
```

Thread `i` and thread `i+2` update the same element `ub[i+1]`. The updates are plain read-modify-write, so updates are lost.
Each statement alone is injective in the loop variable. The three statements together overlap across iterations.
At n <= 32 (one warp, lockstep) the race is hidden. The corpus baseline uses `i_n = 4`, which cannot show it.

## Patch that fixes it (tested, 8 runs at every size)
```julia
CUDA.@atomic ub[i_x - 1] += __oldb_0
CUDA.@atomic ub[i_x] += 2.0 * -__oldb_0
CUDA.@atomic ub[i_x + 1] += __oldb_0
```

## Where to look
The code that decides that a loop is safe to run in parallel and that chooses between a plain update and an atomic update (`cgen_*`, `jgen_*`, `cgen_device_assign`).
Suggested rule: if one array is updated at two or more different index expressions inside one parallel loop body, use atomic updates for all of them, or run the loop serially.

## Second affected kernel: `advection` (tested 2026-10-03, jobs `diag-advection-race2-*`)
The primal is **not** affected. The adjoint has the same defect.

| Item | Result (CUDA and JACC, Float64 and Float32, 5 repeats per case) |
|---|---|
| Primal `advection_cuda` / `advection_jacc` | Float64: error exactly 0 against a sequential CPU reference at all 7 sizes (n = 4 to 1,000,000), zero spread between repeats. Float32: 4e-8 to 5e-7, zero spread. |
| Adjoint, unmodified, `ub` | n = 4 and 33: exact. n >= 100: error 0.06 to 0.8, with different results between repeats (spread up to 0.2). |
| Adjoint, unmodified, scalar gradients `cb`, `dxb`, `dtb` | Wrong as well (7e-3 at n = 100, up to 1 to 2 at n = 1e6), because they read the corrupted `ub` in later reverse steps. |
| Adjoint, unmodified, `u` after call, restored `du`, `dub` | Correct (Float64: exactly 0). Only `ub` and the scalar gradients are affected. |
| Adjoint with atomic `ub` updates | `ub`: 4e-16 to 9e-16 (Float64), 1e-7 to 3e-7 (Float32). Scalar gradients: up to 3e-13 (Float64) and 1e-4 (Float32, n = 1e6, 5e6 accumulated terms). |

Cause: `cuda_kernel_advection_b_4!` and `jacc_kernel_advection_b_4!` contain plain updates:
```julia
ub[i_x] = ub[i_x] + __oldb_2
ub[i_x - 1] = ub[i_x - 1] + -__oldb_2
```
Threads `i` and `i-1` both update `ub[i-1]`. The patch that makes both lines atomic (`CUDA.@atomic` / `Atomix.@atomic`) fixes all sizes.
The Float32 tape (`initstacks_advection_b_*` with `Float32`) allocates and works on CUDA and JACC.

Reference check: the hand-derived CPU reference agrees with finite differences (1e-8 to 1e-11) and with STADE's sequential CPU adjoint (1e-16).
Note for test authors: after the adjoint call, `du` holds its **initial** values, because the reverse sweep restores it from the tape. The first run of the test compared `du` with the final value, which was a flaw in the test and not in STADE.

## Not yet tested
- The static scan of the other generated CUDA adjoints (`dotprod`, `matvec_loss`, `mlp1d`, `mpnn`, `transformer`, `unet`) finds no further case of this pattern. The scan is a heuristic.
- The remaining kernels are not tested for races on the GPU at large sizes. The correctness gate of the benchmark plan will test them.

## Suggested regression test
Run `validate_corpus_gpu.jl` with integer arguments above the warp size (for example `i_n = 1000`) and repeat each adjoint 5 times. A race shows as a wrong or a changing result.

## Verification of the fix (2026-10-04, jobs `verify-fixed-*`)
- Your static test `validate_write_overlap.jl`: 23/23 checks pass in the sandbox.
- Regeneration diff: only the adjoints of `stencil_loss` and `advection` changed (CUDA and JACC). The other adjoints and all primals are byte-identical. The static scan flags 0 kernels (before: 2).
- Stencil adjoint, n = 4 to 1,000,000, 8 repeats per size, CUDA and JACC, Float64: gradient error at most 2.9e-16, identical repeats. (Before the fix: 0.7 to 2.4 at n >= 512 and different answers per run at n = 256.)
- Advection adjoint, 7 sizes, 5 repeats, CUDA and JACC, Float64 and Float32: `ub` error at most 9e-16 (Float64) and 3e-7 (Float32). (Before the fix: 0.06 to 0.8 at n >= 100.) Scalar gradients and the other outputs are correct.
- Single precision against PyTorch and JAX: the STADE float32 error equals the framework float32 error on all cases.
- Limits stated in your code (additive writes only, different index expressions only) remain. Plain-assignment races are not covered. I did not test them.

## Open item: JACC reduction omits one term (pre-existing, not related to the race)
`stencil_loss_jacc` and the forward sweep of `stencil_loss_b_jacc` compute the loss for `n >= 32770` with:
```julia
__jgen_redval_2 = JACC.@parallel_reduce(range = div((i_n - 1) - 2, 1) + 1, (((i_x2, w)->w[i_x2] ^ 2))(w))
```
The source loop is `for i_x2 = 2:i_n-1`. The reduction index starts at 1, so the sum covers `w[1..n-2]` and misses `w[n-1]`.
Observed loss error (Float64, JACC): 9.9e-7 at n = 40000. A CPU calculation of the missing term predicts 9.86e-7. At n = 1,000,000 the prediction is 6.76e-7 and the observation is about 7e-7.
The CUDA reduction `sum(init = zero(eltype(view(w, 2:i_n - 1))), abs2, view(w, 2:i_n - 1))` is correct, and the JACC `jacc_kernel_*` loops that use `__jacc_i` with an explicit offset are correct.
Likely fix: add the loop's lower bound minus one to the reduction index (`w[i_x2 + 1]`), or map the range in the lambda.
Exposed kernels among the benchmark kernels: only `stencil_loss`. `dotprod`, `matvec_loss`, `transformer_loss`, `mpnn_loss`, and `unet_loss` reduce over loops that start at 1.
Suggested test: a JACC primal of a reduction loop that starts at 2, with n above the reduction threshold (32768), compared with a sequential sum.

## Limitation: no GPU primal for a kernel that calls a kernel
`stade_cuda_file` on a root kernel that calls another kernel (for example `mpnn_loss` calling `mpnn`) fails in `cgen_ingest` ("unsupported statement form `Expr(:call, ...)`"). The old and the fixed STADE behave the same. The adjoint path inlines the callee and works.
