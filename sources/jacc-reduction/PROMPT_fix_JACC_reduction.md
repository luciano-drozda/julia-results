# Prompt for the STADE session: fix the JACC idiomatic-reduction path

Copy everything below the line into the other session.

---

You are working on STADE.jl (the version in which you fixed the cross-thread write-overlap data race). A separate benchmarking session found two defects in the **JACC idiomatic-reduction path** of the generator. They live in the same code region, and one of them probably explains the 30 failures you flagged in `test/validate_backend_agreement.jl` (`cuda=N jacc=N+1`, "JACC launch with no zero-trip guard"). Your job: reproduce, fix, test, and report. **Verify every claim below yourself before you rely on it.** The claims come from another session and may contain mistakes.

## Where the code is
All names refer to `src/STADE.jl`. Line numbers are approximate and refer to the fixed version.

- `jgen_idiomatic_reduction_value(arrs, term, loopvar, n_iter)` (about line 6317). Builds the closure for `JACC.@parallel_reduce`. Its comment says that, unlike CUDA/AMDGPU/Metal, `term` "needs no substitution" because "JACC's convention is a closure whose first parameter IS the loop index".
- `jgen_reduction_writeback_kernel` and `jgen_reduction_writeback_launch` (about lines 6330 to 6343). The one-thread kernel and its `JACC.@parallel_for range = 1` launch that fold the reduced value into the target.
- The caller in `jgen_body` (about lines 7285 to 7312): `n_iter = cgen_trip_count(stmt.lo, stmt.step, stmt.hi)`, then `value = jgen_idiomatic_reduction_value(arrs, term, stmt.var, n_iter)`, then `emit_if(n_iter < reduction_threshold, <atomic kernel launch>, <reduce + write-back>)`.
- The CUDA counterpart uses `cgen_reduction_range(lo, step, hi)` and a `view`, so it handles lower bound, stride, and direction. Use it as the reference behavior.

## Defect A: JACC reductions sum the wrong elements (silent wrong value)

**Symptom.** If the reduction loop does not run `1:n` with step 1, the JACC reduce branch returns a wrong sum. CUDA is exact.

**Cause (to confirm).** `jgen_idiomatic_reduction_value` receives only the trip count `n_iter`. `JACC.@parallel_reduce(range = n_iter, (i, arrs...) -> term)` passes `i = 1 .. n_iter`. The closure's first parameter is named like the loop variable, so `u[i_x]` reads elements `1 .. n_iter`. The loop's `lo`, `step`, and direction are lost. The comment above the function is wrong for any loop that does not start at 1 with step 1.

**Why the tests did not see it.** The reduce branch runs only when `n_iter >= reduction_threshold` (default 32768). Corpus sizes are 3 to 5. The same invisibility caused the earlier data race.

**Evidence** (Tesla V100, JACC 1.3.1, Julia 1.11.9, code generated with `reduction_threshold = 0` to force the reduce branch). Four kernels, `loss` starts at 0, `u[i] = 0.5 + sin(0.1 i)`:

| Kernel (loop) | n | Expected | CUDA | JACC | JACC rel. error |
|---|---|---|---|---|---|
| `red_lo2` (`2:i_n`) | 100 | 90.616806 | 90.616806 | 90.974668 | 3.9e-3 |
| `red_lo2` | 1000 | 753.06684 | 753.06684 | 753.4266 | 4.8e-4 |
| `red_step2` (`1:2:i_n`) | 100 | 45.561876 | 45.561876 | 45.99323 | 9.5e-3 |
| `red_step2` | 1000 | 376.78849 | 376.78849 | 376.51497 | 7.3e-4 |
| `red_rev3` (`i_n:-1:3`) | 100 | 90.128667 | 90.128667 | 90.972865 | 9.4e-3 |
| `red_rev3` | 1000 | 752.57871 | 752.57871 | 753.41852 | 1.1e-3 |
| `red_lo3s3` (`3:3:i_n`) | 100 | 30.218308 | 30.218308 | 43.765546 | 4.5e-1 |
| `red_lo3s3` | 1000 | 251.03596 | 251.03596 | 265.21006 | 5.6e-2 |

The generated JACC code for all four uses the lambda `u[i_x] ^ 2` with `range` equal to the trip count, for example:
`JACC.@parallel_reduce(range = div(i_n - 2, 1) + 1, (((i_x, u)->u[i_x] ^ 2))(u))` for `2:i_n`.
A related observation from `stencil_loss` (loop `2:i_n-1`, default threshold, n = 40000): the JACC loss is off by 9.9e-7. A CPU calculation of "sum over `w[1..n-2]` instead of `w[2..n-1]`" predicts 9.86e-7. At n = 1,000,000 it predicts 6.76e-7, and the observation is about 7e-7.

The four kernels (skill-stade style):

```julia
function red_lo2(u, loss, i_n)
    for i_x = 2:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
function red_step2(u, loss, i_n)
    for i_x = 1:2:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
function red_rev3(u, loss, i_n)
    for i_x = i_n:-1:3
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
function red_lo3s3(u, loss, i_n)
    for i_x = 3:3:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
```

Generate the primal with `STADE.stade_jacc_file(path, out; reduction_threshold = 0)` and `STADE.stade_cuda_file(...)` with the same keyword. (Full archived files: branch `bench-raw` of `github.com/luciano-drozda/julia-results`, folder `sources/jacc-reduction/`, if you can reach it.)

**Requested fix.**
1. Make the JACC closure independent of the position index: use a fresh index name (for example `__jgen_k`) as the first closure parameter, and substitute `loopvar -> lo + (__jgen_k - 1) * step` in `term` (and in every array index it contains). Handle negative steps and non-literal `lo` and `step` expressions. Keep the existing `max(0, ...)` style of trip count.
2. When `lo == 1` and `step == 1`, the emitted code must stay **byte-identical** to the current output. `dotprod`, `matvec_loss`, and the loss wrappers must not change.
3. Correct the misleading comment on `jgen_idiomatic_reduction_value`.
4. Check whether any other caller or mode (tangent, HVP, other backends) builds a `@parallel_reduce` closure the same way. Report what you find. Do not change what you cannot test.

## Defect B: no zero-trip guard around the JACC reduce branch

**Symptom.** With `reduction_threshold = 0`, a zero-trip reduction loop makes the JACC code raise an exception. CUDA returns 0.

**Evidence** (same GPU run, `n` chosen so that the trip count is 0: `red_lo2` with `i_n = 1`, `red_step2` with `i_n = 0`, `red_rev3` and `red_lo3s3` with `i_n = 2`): all four JACC cases raise
`Grid dimensions CuDim3(0x00000000, 0x00000001, 0x00000001) are not positive`. All four CUDA cases return 0.

**Cause (to confirm).** The branch is `emit_if(n_iter < reduction_threshold, <atomic path, guarded>, <reduce path>)`. With threshold 0, the condition `n_iter < 0` is false for `n_iter = 0`, so the reduce path runs with `range = 0` and no `> 0` guard. At the default threshold (32768) the reduce path needs `n_iter >= 32768`, so the defect is latent. Look at the generated code for `stencil_loss` with `reduction_threshold = 0`:

```julia
if div((i_n - 1) - 2, 1) + 1 < 0
    if div((i_n - 1) - 2, 1) + 1 > 0
        JACC.@parallel_for range = ... jacc_kernel_stencil_loss_3!(i_n, loss, w)
    end
else
    __jgen_redval_2 = JACC.@parallel_reduce(range = div((i_n - 1) - 2, 1) + 1, ...)
    JACC.@parallel_for range = 1 jacc_kernel_stencil_loss_2!(loss, __jgen_redval_2)
end
```

**Requested fix.** Wrap the reduce and its write-back in a `n_iter > 0` guard (use the same expression as the other guards in the file).

## The 30 failures in `validate_backend_agreement.jl`

The other session reproduced them in the sandbox (no GPU needed), with the previous source and with the race-fixed source. Both give the same 112/142 result:

- All 30 are in `:adjoint` mode. None are in `:hvp`.
- Every message has the form `device kernel count: cuda=N jacc=N+1` and `1 JACC launch(es) with no zero-trip guard`.
- Check done by the other session: the JACC adjoint output of **all 30** failing kernels contains `JACC.@parallel_reduce`, and **none of the 41** passing adjoint kernels does. So the failures are exactly the kernels that use the idiomatic reduction path. Re-run this check yourself.
- Hypothesis for the cause (confirm or refute): the only unguarded launch is the write-back `JACC.@parallel_for range = 1 ...`. CUDA folds the reduced value on the host (`CUDA.@allowscalar loss[1] = loss[1] + sum(...)`), so JACC has one more device kernel by design. The failures appeared when the idiomatic-reduction write-back was added after the test was written.
- Note: `range = 1` can never be a zero-trip launch. So the property check flags a harmless launch, while the real zero-trip hazard (Defect B) sits in the reduce call, which the check does not inspect.

**What to do.**
1. Fix Defect B. After that, the write-back launch sits inside a `> 0` guard, which may remove the "no zero-trip guard" message.
2. For the kernel count, decide on a rule and write it into the test and its header: either count the CUDA host fold as a kernel, or exclude write-back kernels from the comparison. State why.
3. **Do not relax the test just to make it pass.** After your change, remove the new guard on purpose and confirm that the test fails (a sabotage check, as the test already does for other guards). Add a property check that a `JACC.@parallel_reduce` also sits inside a `> 0` guard.

## Tests to add or run

1. Static (no GPU): for the four kernels above, generated with `reduction_threshold = 0`, assert that the JACC closure does not use the bare loop variable as an index unless `lo == 1` and `step == 1`. Assert that CUDA output is unchanged.
2. Regeneration diff: regenerate every corpus kernel (adjoint and HVP, CUDA and JACC) before and after your change. Report which files differ. Only kernels with an idiomatic reduction and a non-unit `lo` or `step` may change.
3. Numeric (needs a GPU): for the four kernels, JACC must equal CUDA to 1e-12 relative at n = 100 and n = 1000, and the zero-trip cases must return 0 with no exception. Use `reduction_threshold = 0`. Also run the existing corpus reduction kernels whose loop is not `1:i_n` (`red_strided` has step 2 and `partialdot` starts at 2) through `validate_corpus_gpu.jl` with `reduction_threshold = 0` on JACC, if the harness allows that keyword. Two kernels look similar but do **not** expose Defect A: `red_reverse` runs `i_n:-1:1`, so it sums the same set of elements in another order, and `sumsq_shifted` runs `1:i_n` with the shift inside the term. Use them only as controls: they must stay correct and unchanged.
4. Run `test/validate_backend_agreement.jl`, `test/validate_write_overlap.jl`, and every other test that needs no GPU. Report the totals.
5. Bump the version string (patch level), so that benchmark results can name the exact source.

## What to report back
- For each defect: confirmed, refuted, or partly confirmed, with your own evidence.
- The code change (diff summary) and the corrected comment.
- The regeneration diff: which generated files changed and which did not.
- Test results, including the sabotage check and the final state of `validate_backend_agreement.jl`.
- Anything you could not test, and any other place where the same pattern (index position instead of loop value) may occur.
- The new SHA-256 of `src/STADE.jl`.

Constraints: do not change CUDA output. Do not weaken or delete existing checks. Keep the code comments in the style of the file. Do not hide a failure in a test by narrowing its scope without saying so.
