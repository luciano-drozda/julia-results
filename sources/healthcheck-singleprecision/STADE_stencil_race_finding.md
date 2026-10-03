# STADE 0.4.3: data race in the generated GPU adjoint of `stencil_loss`

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

## Not yet tested
- `advection`: a static scan of the generated CUDA adjoints finds the same pattern in `cuda_kernel_advection_b_4!` (`ub[i_x]` and `ub[i_x - 1]`, plain updates). It is **not tested on the GPU**.
- The scan is a heuristic. It checks only plain updates of one shadow array at several indices.

## Suggested regression test
Run `validate_corpus_gpu.jl` with integer arguments above the warp size (for example `i_n = 1000`) and repeat each adjoint 5 times. A race shows as a wrong or a changing result.
