# STADE 0.4.3: data race in the generated GPU adjoints of `stencil_loss` and `advection`

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
