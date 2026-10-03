# Single-precision health check and stencil_loss data-race diagnosis (2026-10-03)

Jobs (raw results in raw/healthchecks/): healthcheck-sp-v100-*, diag-stencil-race-* (patch did NOT apply, ignore the patched column), diag-stencil-race2-* (valid).
Finding: the generated CUDA/JACC adjoint of stencil_loss updates ub[i-1], ub[i], ub[i+1] with plain read-modify-write in parallel threads (data race).
Exact for n <= 100, intermittent at n = 256, wrong for n >= 512. Making the three updates atomic gives max relative error 2.9e-16 at all sizes.
STADE 0.4.3, Julia 1.10.11 generation, Julia 1.11.9 execution, Tesla V100-PCIE-16GB, krakengpu1.
