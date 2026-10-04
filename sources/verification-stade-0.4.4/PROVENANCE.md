# STADE 0.4.4 verified 2026-10-04 (V100, krakengpu1, Julia 1.11.9; generation with Julia 1.10.11)
The STADE source is NOT stored here (public branch). Only hashes.

| File | SHA-256 |
|---|---|
| STADE_jl.zip (0.4.4) | c5dd3d9904d0386e312cee31c4b05b3c92d4b87d73980a16667cec3d1f53278a |
| 0.4.4 src/STADE.jl | ca53962b474b087302fe9792249b07707ab1000f2d1f2ceffa9ad04c723dceab |
| 0.4.3 + race fix src/STADE.jl | 1736cb77d1cc1e7d5a67b6e092ca81fcf8b78b7ca0149bda9398dbf3d3a22f4f |

Static tests (sandbox, no GPU): validate_backend_agreement 142/142 (was 112/142), validate_jacc_reduction 38/38, validate_write_overlap 23/23.
Generated code vs 0.4.3+race fix: CUDA files all identical. JACC files changed only where an idiomatic reduction exists (dotprod, matvec_loss, stencil_loss, mpnn_loss, transformer_loss, unet_loss). advection, mlp1d, mpnn, transformer, unet JACC identical.
GPU jobs (raw results in raw/healthchecks/): verify044-jacc-reduction-* (36/36 cases ok), verify044-stencil-sweep-* (<= 2.9e-16), verify044-singleprecision-*, verify044-dot-matvec-* (<= 1.9e-14).
Not re-run on GPU: advection and the race-fixed kernels (generated files byte-identical to the verified 0.4.3 + race-fix output).
