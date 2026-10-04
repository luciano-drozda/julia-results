#!/usr/bin/env python3
"""Analysis of raw benchmark results (plan sections 6, 8, 13).
  analyze.py report  <rawdir> <outdir>                      threshold-free tables, ratios, performance profile; bands only if frozen
  analyze.py freeze-thresholds <outdir> --runtime a,b,c --memory a,b   write analysis/thresholds.json with date and SHA-256 BEFORE band counts
Input: every result.json under <rawdir> (layout raw/<phase>/<job>/result.json). The script never modifies raw files."""
import sys, os, json, glob, hashlib, datetime, math, argparse, collections
import numpy as np

STADE = ("S-CUDA", "S-JACC", "S-CPU")
FRAMEWORKS = ("P-EAGER", "P-GRAPH", "P-COMPILE", "J-JIT", "J-NN")
DEFAULT_FW = ("P-EAGER", "J-JIT")

def load_results(rawdir):
    out = []
    for p in sorted(glob.glob(os.path.join(rawdir, "**", "result.json"), recursive=True)):
        try: d = json.load(open(p))
        except Exception: continue
        if isinstance(d, dict) and "records" in d: d["_path"] = p; out.append(d)
    return out

def med(v): return float(np.median(v))

def summarize(vals):
    v = np.asarray(vals, dtype=float)
    return {"med": float(np.median(v)), "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)), "min": float(v.min()),
            "cv": float(v.std() / v.mean()) if v.size > 1 and v.mean() > 0 else 0.0, "n": int(v.size)}

def key_of(r): return (r.get("case"), r.get("size"), r.get("mode"), r.get("contender"))

def collect(results):
    """time: key -> list of per-record medians (one per pass per job). mem, gate, train likewise."""
    time_, gate, mem, train = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    hosts = collections.defaultdict(set)
    for d in results:
        for r in d["records"]:
            k = r.get("kind")
            if k == "time" and r.get("trial_us"):
                time_[key_of(r)].append(med(r["trial_us"])); hosts[key_of(r)].add(d.get("host"))
            elif k == "gate": gate[key_of(r)].append({"max_err": r.get("max_err"), "passed": r.get("passed"), "det": r.get("determinism_err"), "job": d.get("job")})
            elif k == "mem": mem[key_of(r)].append(r)
            elif k == "train":
                train[(r.get("model"), r.get("size"), r.get("contender"))].append(r)
    return time_, gate, mem, train, hosts

def time_table(time_):
    rows = []
    for k, v in sorted(time_.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        s = summarize(v); rows.append(dict(case=k[0], size=k[1], mode=k[2], contender=k[3], **s))
    return rows

def ratios(rows):
    """R = STADE median time / best framework median time, per (case, size, mode). Frameworks with scope 'all' only."""
    by = collections.defaultdict(dict)
    for r in rows: by[(r["case"], r["size"], r["mode"])][r["contender"]] = r["med"]
    out = []
    for key, d in sorted(by.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        fw = {c: t for c, t in d.items() if c in FRAMEWORKS}
        if not fw: continue
        best_c, best_t = min(fw.items(), key=lambda x: x[1])
        dflt = {c: d[c] for c in DEFAULT_FW if c in d}
        for s in STADE:
            if s not in d: continue
            row = dict(case=key[0], size=key[1], mode=key[2], stade=s, t_stade_us=d[s], best_framework=best_c, t_best_us=best_t, R=d[s] / best_t)
            for c, t in dflt.items(): row["R_vs_" + c] = d[s] / t
            out.append(row)
    return out

def band(R, th):
    f, c, s = th
    return "faster" if R < f else ("competitive" if R <= c else ("slower" if R <= s else "far slower"))

def performance_profile(Rs, grid=None):
    grid = grid if grid is not None else np.round(np.concatenate([np.arange(0.5, 2.0, 0.1), np.arange(2.0, 20.01, 0.5)]), 3)
    Rs = np.asarray(Rs, dtype=float)
    return [(float(x), float(np.mean(Rs <= x)) if Rs.size else 0.0) for x in grid]

def freeze(outdir, runtime, memory):
    os.makedirs(outdir, exist_ok=True)
    th = {"runtime": {"faster": runtime[0], "competitive": runtime[1], "slower": runtime[2]},
          "memory": {"equal": memory[0], "competitive": memory[1]},
          "frozen_utc": datetime.datetime.utcnow().isoformat() + "Z"}
    body = json.dumps(th, indent=1, sort_keys=True); path = os.path.join(outdir, "thresholds.json")
    open(path, "w").write(body); h = hashlib.sha256(body.encode()).hexdigest()
    open(os.path.join(outdir, "thresholds.sha256"), "w").write(h + "\n")
    return path, h

def load_thresholds(outdir):
    p, hp = os.path.join(outdir, "thresholds.json"), os.path.join(outdir, "thresholds.sha256")
    if not (os.path.exists(p) and os.path.exists(hp)): return None
    body = open(p).read()
    if hashlib.sha256(body.encode()).hexdigest() != open(hp).read().strip(): raise SystemExit("thresholds.json does not match thresholds.sha256 (edited after freezing)")
    return json.loads(body)

def mem_table(mem):
    rows = []
    for k, recs in sorted(mem.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        ma = [r.get("M_A") for r in recs if r.get("M_A") is not None]
        mb = [r.get("M_B") for r in recs if r.get("M_B") is not None]; mc = [r.get("M_C") for r in recs if r.get("M_C") is not None]
        rows.append(dict(case=k[0], size=k[1], mode=k[2], contender=k[3], M_A=med(ma) if ma else None, M_B=med(mb) if mb else None, M_C=med(mc) if mc else None, n=len(recs)))
    return rows

def mem_ratios(rows):
    by = collections.defaultdict(dict)
    for r in rows: by[(r["case"], r["size"], r["mode"])][r["contender"]] = r["M_A"]
    out = []
    for key, d in by.items():
        fw = {c: m for c, m in d.items() if c in FRAMEWORKS and m}
        if not fw: continue
        bc, bm = min(fw.items(), key=lambda x: x[1])
        for s in STADE:
            if d.get(s): out.append(dict(case=key[0], size=key[1], mode=key[2], stade=s, M_A_stade=d[s], best_framework=bc, M_A_best=bm, ratio=d[s] / bm))
    return out

def write_csv(path, rows):
    if not rows: open(path, "w").write(""); return
    cols = list(rows[0].keys()); 
    for r in rows:
        for c in r:
            if c not in cols: cols.append(c)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows: f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")

def report(rawdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    results = load_results(rawdir); time_, gate, mem, train, hosts = collect(results)
    tt = time_table(time_); R = ratios(tt); mt = mem_table(mem); mr = mem_ratios(mt)
    write_csv(os.path.join(outdir, "time_summary.csv"), tt); write_csv(os.path.join(outdir, "ratios.csv"), R)
    write_csv(os.path.join(outdir, "memory_summary.csv"), mt); write_csv(os.path.join(outdir, "memory_ratios.csv"), mr)
    gate_rows = [dict(case=k[0], size=k[1], mode=k[2], contender=k[3], worst_err=max([g["max_err"] for g in v if g["max_err"] is not None] or [None]) if any(g["max_err"] is not None for g in v) else None,
                      all_passed=all(g["passed"] for g in v if g["passed"] is not None) if any(g["passed"] is not None for g in v) else None,
                      worst_determinism=max([g["det"] for g in v if g["det"] is not None] or [None]) if any(g["det"] is not None for g in v) else None, n=len(v)) for k, v in sorted(gate.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]
    write_csv(os.path.join(outdir, "gate_summary.csv"), gate_rows)
    unstable = [dict(case=r["case"], size=r["size"], mode=r["mode"], contender=r["contender"], cv=r["cv"], n=r["n"]) for r in tt if r["n"] > 1 and r["cv"] > 0.10]
    write_csv(os.path.join(outdir, "unstable_cases.csv"), unstable)
    tr_rows = []
    for k, recs in sorted(train.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        st = [med(r["sync_step_us"]) for r in recs if r.get("sync_step_us")]; tp = [r["throughput_steps_per_s"] for r in recs if r.get("throughput_steps_per_s")]
        tr_rows.append(dict(model=k[0], size=k[1], contender=k[2], sync_step_us_med=med(st) if st else None, throughput_steps_per_s=med(tp) if tp else None, n=len(recs),
                            loss_1=recs[0]["losses_at"].get("1"), loss_300=recs[0]["losses_at"].get("300")))
    write_csv(os.path.join(outdir, "training_summary.csv"), tr_rows)
    # threshold-free performance profile (always)
    prof = {}
    for mode in ("primal", "adjoint"):
        for s in STADE:
            rs = [r["R"] for r in R if r["mode"] == mode and r["stade"] == s]
            if rs: prof[f"{mode}/{s}"] = performance_profile(rs)
    json.dump(prof, open(os.path.join(outdir, "performance_profile.json"), "w"))
    summary = {"jobs": len(results), "time_keys": len(tt), "ratio_rows": len(R), "geomean_R": {}}
    for mode in ("primal", "adjoint"):
        for s in STADE:
            rs = [r["R"] for r in R if r["mode"] == mode and r["stade"] == s]
            if rs: summary["geomean_R"][f"{mode}/{s}"] = float(np.exp(np.mean(np.log(rs)))); summary.setdefault("n_cases", {})[f"{mode}/{s}"] = len(rs)
    th = load_thresholds(outdir)
    if th is None:
        summary["bands"] = "NOT COMPUTED: thresholds are not frozen (run freeze-thresholds first)"
    else:
        t = (th["runtime"]["faster"], th["runtime"]["competitive"], th["runtime"]["slower"]); bands = collections.defaultdict(collections.Counter)
        for r in R: r["band"] = band(r["R"], t); bands[f"{r['mode']}/{r['stade']}"][r["band"]] += 1
        write_csv(os.path.join(outdir, "ratios_with_bands.csv"), R)
        summary["bands"] = {k: dict(v) for k, v in bands.items()}; summary["thresholds"] = th
        sens = {}
        for name, tt_ in (("1.5/3.0", (1.0, 1.5, 3.0)), ("3.0/10.0", (1.0, 3.0, 10.0))):
            c2 = collections.defaultdict(collections.Counter)
            for r in R: c2[f"{r['mode']}/{r['stade']}"][band(r["R"], tt_)] += 1
            sens[name] = {k: dict(v) for k, v in c2.items()}
        summary["sensitivity"] = sens
        mb = collections.Counter()
        for r in mr: mb[("equal" if r["ratio"] <= th["memory"]["equal"] else "competitive" if r["ratio"] <= th["memory"]["competitive"] else "larger")] += 1
        summary["memory_bands"] = dict(mb)
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for key, pts in prof.items():
            plt.plot([p[0] for p in pts], [p[1] for p in pts], label=key)
        plt.xscale("log"); plt.xlabel("R = STADE time / best framework time"); plt.ylabel("fraction of cases with R <= x"); plt.legend(fontsize=7); plt.grid(alpha=.3)
        plt.savefig(os.path.join(outdir, "performance_profile.png"), dpi=130); plt.close()
    except Exception as e:
        summary["plot_error"] = repr(e)[:100]
    return summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("report"); a.add_argument("rawdir"); a.add_argument("outdir")
    f = sub.add_parser("freeze-thresholds"); f.add_argument("outdir"); f.add_argument("--runtime", default="1.0,2.0,5.0"); f.add_argument("--memory", default="1.0,1.5")
    args = ap.parse_args()
    if args.cmd == "report": print(json.dumps(report(args.rawdir, args.outdir), indent=1))
    elif args.cmd == "freeze-thresholds":
        p, h = freeze(args.outdir, [float(x) for x in args.runtime.split(",")], [float(x) for x in args.memory.split(",")]); print("frozen:", p, h)
    else: ap.print_help()
