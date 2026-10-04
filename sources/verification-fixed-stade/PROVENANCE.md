# Fixed STADE 0.4.3, verified 2026-10-04 (V100, krakengpu1, Julia 1.11.9; generation with Julia 1.10.11)

The STADE source is NOT stored here (this branch is public). Only hashes are recorded.

| File | SHA-256 |
|---|---|
| STADE_jl.zip (uploaded fixed version) | d60e495af3bc42451232aeca068368ffdba6a07bf7b0d874795302df1fd52d51 |
| fixed src/STADE.jl | 1736cb77d1cc1e7d5a67b6e092ca81fcf8b78b7ca0149bda9398dbf3d3a22f4f |
| previous src/STADE.jl (with the data race) | 96d95303c5d50fe2392db2a9040cda1702016acb32bfc26fc341ed0ffca0677f |

Changes found by diff: new cross-write overlap analysis (cgen_race_write_arrays) used by the CUDA and JACC generators; new test test/validate_write_overlap.jl; validate_corpus_gpu.jl gets int_lo, int_hi, grow_max arguments.
Static test test/validate_write_overlap.jl: 23/23 checks passed (run in the sandbox).
Generated adjoints that changed: stencil_loss and advection only (CUDA and JACC, 4 to 6 lines each). All other adjoints and all primals are byte-identical.

GPU verification jobs (raw results in raw/healthchecks/): verify-fixed-stencil-sweep-*, verify-fixed-advection-*, verify-fixed-singleprecision-*.
Result: stencil gradient error <= 2.9e-16 for n = 4 ... 1,000,000 (8 identical repeats per size, CUDA and JACC). advection ub error <= 9e-16 (Float64), 3e-7 (Float32), zero spread in Float64.
Open item (separate, pre-existing): JACC reduction path of stencil_loss omits one term when the loop starts at 2 (loss error 9.9e-7 at n = 40000, 6.8e-7 predicted at 1e6; prediction from a CPU calculation matched to two digits).
