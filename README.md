# bench-raw: raw results of the STADE vs PyTorch vs JAX benchmark

This branch holds raw data only. It holds no conclusions and no thresholds.
The relay branch `main` is cleared at the start of every session. This branch is not.

## Layout

    raw/<phase>/<job-name>/result.json      what the job printed (one JSON line)
    raw/<phase>/<job-name>/job.json         the payload that ran (script and inputs)
    raw/<phase>/<job-name>/submission.json  the Slurm job id
    MANIFEST.sha256                         SHA-256 of every file on this branch
    sources/                                generated STADE sources, harnesses, pip lists (added later)
    analysis/                               analysis scripts and thresholds.json (added later)

## Phases

- `healthchecks`: environment checks (not benchmark data)
- `p1-preflight`: job J0
- `p2-timing`: jobs J1 to J5
- `p3-training-memory`: jobs J6 and J7

## Rules

1. Never edit a raw file. Add new files only.
2. Verify with: `sha256sum -c MANIFEST.sha256`
3. The repository is public. Everything here is world-readable.
