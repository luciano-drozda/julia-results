# JACC reduction defects (2026-10-04, V100, fixed-race STADE, JACC 1.3.1)
Job diag-jacc-reduction-* failed because of a module-name collision in MY driver (not STADE). Job diag-jacc-reduction2-* is the valid run.
A) JACC idiomatic reduction ignores the loop lower bound / step / direction: wrong value for red_lo2, red_step2, red_rev3, red_lo3s3 (CUDA exact).
B) With reduction_threshold = 0 the JACC reduce branch has no `> 0` guard: zero trip count raises "Grid dimensions ... are not positive" (CUDA fine).
validate_backend_agreement.jl: the same 30 failures with the race-fixed and the previous STADE source (so not caused by the race fix).
