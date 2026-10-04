"""Shared helpers for the benchmark harnesses: deterministic data, signatures, statistics.
The data generator uses only exact integer arithmetic and IEEE operations, so Julia and Python
produce bit-identical arrays from the same spec."""
import json, math
import numpy as np

MASK = (1 << 64) - 1
SPEC = None

def load_spec(path_or_text):
    global SPEC
    SPEC = json.loads(path_or_text) if path_or_text.lstrip().startswith("{") else json.load(open(path_or_text))
    return SPEC

def _unif_bits(n, seed):
    """n values of 64-bit splitmix output for indices 0..n-1 and an integer seed."""
    i = np.arange(n, dtype=np.uint64)
    base = np.uint64((seed * 0xD1B54A32D192ED03 + 0x632BE59BD9B4E019) & MASK)
    z = i * np.uint64(0x9E3779B97F4A7C15) + base
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))

def unif(n, seed):
    return (_unif_bits(n, seed) >> np.uint64(11)).astype(np.float64) * 2.0 ** -53

def ev(expr, env):
    return eval(str(expr), {"__builtins__": {}, "sqrt": math.sqrt}, dict(env))

def to_int(x): return int(round(ev(x, {}) if isinstance(x, str) else x))

def fam_env(fam, params):
    env = dict(params)
    for k, v in (SPEC["families"][fam].get("derived") or {}).items():
        env[k] = int(round(ev(v, env)))
    return env

def make_data(fam, params, which="adjoint"):
    """Returns (arrays, scalars, ints). Arrays are float64 (F-order for 2-D) or int64 (1-based indices)."""
    F = SPEC["families"][fam]; env = fam_env(fam, params); base = SPEC["seed_base"]
    arrays, scalars, ints = {}, {}, {}
    for a in list(F["args"]) + list(F.get("extra") or []):
        nm, kind = a["name"], a["kind"]
        if kind == "int": ints[nm] = int(round(ev(a["value"], env))); continue
        if kind == "fscalar": scalars[nm] = float(ev(a["value"], env)); continue
        n = int(round(ev(a["len"], env))); seed = a.get("seed", 0) * 7919 + base
        if kind == "out": arr = np.zeros(n)
        elif kind == "in":
            scale = float(ev(a.get("scale", "1"), env)); shift = float(ev(a.get("shift", "0"), env))
            arr = scale * (2.0 * unif(n, seed) - 1.0) + shift
        elif kind == "idx":
            rng = int(round(ev(a["range"], env)))
            arr = ((_unif_bits(n, seed) >> np.uint64(11)) % np.uint64(rng)).astype(np.int64) + 1
        if a.get("shape"):
            shp = tuple(int(round(ev(s, env))) for s in a["shape"]); arr = arr.reshape(shp, order="F")
        arrays[nm] = arr
    return arrays, scalars, ints

def grad_names(fam, scope="all"):
    """Names of the float inputs whose gradients are compared. scope 'params' keeps parameters only."""
    F = SPEC["families"][fam]; out = []
    for a in F["args"]:
        if a.get("grad") and (scope == "all" or a.get("param")): out.append(a["name"])
    return out

# ---- signatures: a compact fingerprint of an array that works across processes and languages
def sig_positions(n):
    if n <= 1024: return np.arange(n)
    return (_unif_bits(1024, 999) % np.uint64(n)).astype(np.int64)

def sig(a):
    a = np.asarray(a, dtype=np.float64)
    flat = a.reshape(-1, order="F") if a.ndim > 1 else a.reshape(-1)
    n = flat.size
    if n == 0: return {"n": 0, "sum": 0.0, "sumsq": 0.0, "maxabs": 0.0, "samples": []}
    return {"n": int(n), "sum": float(flat.sum()), "sumsq": float((flat * flat).sum()), "maxabs": float(np.abs(flat).max()),
            "samples": [float(v) for v in flat[sig_positions(n)]]}

def sig_err(a, b, floor=0.0):
    """Relative difference between two signatures (a: candidate, b: reference): the largest of four measures.
    `floor` is an absolute scale below which differences are treated as rounding noise. It matters for
    gradients that are analytically zero (for example the key bias of attention, which softmax cancels)."""
    if a["n"] != b["n"]: return float("inf")
    if b["n"] == 0: return 0.0
    fl = max(floor, 1e-300); n = b["n"]
    e_sq = abs(a["sumsq"] - b["sumsq"]) / max(b["sumsq"], n * fl * fl)
    e_sum = abs(a["sum"] - b["sum"]) / max(math.sqrt(n * b["sumsq"]), n * fl)
    sa, sb = np.array(a["samples"]), np.array(b["samples"])
    e_smp = float(np.abs(sa - sb).max() / max(b["maxabs"], fl)) if sa.size else 0.0
    e_max = abs(a["maxabs"] - b["maxabs"]) / max(b["maxabs"], fl)
    return max(e_sq, e_sum, e_smp, e_max)

def noise_floor(ref_flat, rel=1e-8):
    """Absolute floor for comparing the entries of one evaluation: `rel` times the largest gradient magnitude."""
    scales = [v["maxabs"] for k, v in ref_flat.items() if k != "loss" and v and v.get("n", 0) > 0]
    return rel * max(scales) if scales else 0.0

def compare_flat(cand, ref, analytic_zero=()):
    """Errors of candidate entries against reference entries. Entries listed in analytic_zero have an exactly
    zero gradient: the error is the larger magnitude of the two, relative to the scale of the other gradients."""
    scale = max([v["maxabs"] for k, v in ref.items() if k != "loss" and k not in analytic_zero and v and v.get("n", 0) > 0] or [0.0])
    fl = 1e-8 * scale; errs = {}
    for k, v in cand.items():
        if k not in ref or v is None or ref[k] is None: continue
        if k in analytic_zero: errs[k] = max(v["maxabs"], ref[k]["maxabs"]) / max(scale, 1e-300)
        else: errs[k] = sig_err(v, ref[k], fl)
    return errs

def stats(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {"med": float(np.median(v)), "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
            "min": float(v.min()), "mean": float(v.mean()), "n": int(v.size)}


# ---- training data: N synthetic samples per per-sample array, contiguous, generated like the other arrays
def make_dataset(fam, params, N):
    F = SPEC["families"][fam]; env = fam_env(fam, params); base = SPEC["seed_base"]; ds = {}
    for a in F["args"]:
        if a["name"] not in F["per_sample"]: continue
        n = int(round(ev(a["len"], env)))
        scale = float(ev(a.get("scale", "1"), env)); shift = float(ev(a.get("shift", "0"), env))
        seed = a["seed"] * 7919 + base + 1000003
        ds[a["name"]] = (scale * (2.0 * unif(N * n, seed) - 1.0) + shift, n)
    return ds
