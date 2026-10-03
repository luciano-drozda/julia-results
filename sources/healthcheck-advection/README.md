# advection: is the generated GPU code race-free? (2026-10-03, V100, STADE 0.4.3)
Result (run 2, valid): PRIMAL is exact in Float64 (CUDA and JACC) at all sizes with zero run-to-run spread. ADJOINT has the same data race as stencil_loss
(cuda/jacc_kernel_advection_b_4!: plain ub[i_x] and ub[i_x-1] updates). Wrong ub for n >= 100, different results between runs. Atomic patch fixes it.
