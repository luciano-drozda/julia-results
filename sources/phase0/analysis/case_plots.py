#!/usr/bin/env python3
"""Slide plots for one case family from raw result JSON files (generalizes unet_plots.py).
Usage: case_plots.py <dir with result json> <output dir> <K7|K8>
Timing and training records are kept only from passes that recorded no foreign GPU process. No band or 'competitive' label is drawn."""
import sys, os, re, json, glob, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"S-CUDA": "#c0392b", "S-JACC": "#e67e22", "P-EAGER": "#2471a3", "P-GRAPH": "#17a589", "J-JIT": "#7d3c98"}
LAB = {"S-CUDA": "STADE (CUDA)", "S-JACC": "STADE (JACC)", "P-EAGER": "PyTorch eager", "P-GRAPH": "PyTorch CUDA graph", "J-JIT": "JAX jit"}
MRK = {"S-CUDA": "o", "S-JACC": "s", "P-EAGER": "^", "P-GRAPH": "v", "J-JIT": "D"}
ORDER = ["S-CUDA", "S-JACC", "P-EAGER", "P-GRAPH", "J-JIT"]; FW = ("P-EAGER", "P-GRAPH", "J-JIT")
CFG = {
  "K7": dict(model="M3", name="U-Net", prefix="narrow", extra=("wide",), xlabel="image side h = w (pixels)", log=2, ticks=[16, 32, 64, 128, 256], tl=["16", "32", "64", "128", "256"],
             train_title="U-Net training, batch size 1, plain SGD (32×32 input, 3→8→16→32 channels)", ratio_title="Ratio to the best framework (narrow net)", mem_title="Memory (arrays + tape + temporaries)", mem_lab=lambda s: s.replace("narrow-", "n-").replace("wide-", "w-")),
  "K8": dict(model="M4", name="Message-passing network", prefix="nn", extra=(), xlabel="number of nodes (8 edges per node)", log=10, ticks=[1000, 10000, 100000], tl=["1k", "10k", "100k"],
             train_title="Message-passing network training, batch size 1, plain SGD (1000 nodes, 8000 edges)", ratio_title="Ratio to the best framework", mem_title="Memory (arrays + tape + temporaries)", mem_lab=lambda s: {"nn1000": "1k", "nn10000": "10k", "nn100000": "100k"}.get(s, s))}
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "legend.fontsize": 13, "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.grid": True, "grid.alpha": .3})

def xval(sid): return int(re.findall(r"\d+", sid)[-1])
def skey(c): return lambda s: (s.split("-")[0] if "-" in s else "", xval(s))

def load(d):
    recs, meta = [], {"dropped": collections.Counter(), "jobs": [], "hosts": set()}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try: j = json.load(open(p))
        except Exception: continue
        if not isinstance(j, dict) or "records" not in j: continue
        meta["jobs"].append(j.get("job")); meta["hosts"].add(j.get("host")); meta["gpu"] = (j.get("snap_start", {}).get("gpu", "") or "").split(",")[0] or meta.get("gpu")
        ok = {i: (ps.get("cotenant") is not None and ps["cotenant"].get("foreign_processes_after_wait", 1) == 0) for i, ps in enumerate(j.get("passes", []), start=1)}
        for r in j["records"]:
            if r.get("kind") in ("time", "train") and not ok.get(r.get("pass"), False): meta["dropped"][(j.get("job"), r.get("kind"))] += 1; continue
            r["_job"] = j.get("job"); recs.append(r)
    return recs, meta

def times(recs, case, mode):
    out = {}
    for r in recs:
        if r.get("kind") == "time" and r.get("case") == case and r.get("mode") == mode and r.get("scope", "all") == "all" and r.get("trial_us"):
            v = np.asarray(r["trial_us"], float); out[(r["contender"], r["size"])] = (np.median(v), np.percentile(v, 25), np.percentile(v, 75))
    return out

def footer(fig, meta, extra="", what="median of 30 trials"):
    fig.text(0.01, 0.008, f"{meta.get('gpu') or 'GPU'} · float64 · batch 1 · {what}, bars = quartiles · ONE repeat, no stability check{(' · ' + extra) if extra else ''}", fontsize=10.5, color="#555")

def setx(ax, c):
    ax.set_xscale("log", base=c["log"]); ax.set_xticks(c["ticks"]); ax.set_xticklabels(c["tl"]); ax.minorticks_off(); ax.set_xlabel(c["xlabel"])

def panel_time(ax, T, title, c):
    for k in ORDER:
        for fam, ls, fill in [(c["prefix"], "-", True)] + [(e, "", False) for e in c["extra"]]:
            pts = sorted([(xval(s), v) for (cc, s), v in T.items() if cc == k and s.startswith(fam)])
            if not pts: continue
            x = [p[0] for p in pts]; y = [p[1][0] for p in pts]; lo = [p[1][0] - p[1][1] for p in pts]; hi = [p[1][2] - p[1][0] for p in pts]
            ax.errorbar(x, y, yerr=[lo, hi], color=COL[k], marker=MRK[k], ms=8, lw=2.2 if ls else 0, ls=ls or "none", mfc=COL[k] if fill else "white", mec=COL[k], mew=2, capsize=3, label=LAB[k] if fam == c["prefix"] else None)
    setx(ax, c); ax.set_yscale("log"); ax.set_ylabel("time per call (µs)"); ax.set_title(title)

def panel_ratio(ax, TP, TA, c):
    for mode, T, ls in (("adjoint", TA, "-"), ("primal", TP, "--")):
        for k in ("S-CUDA", "S-JACC"):
            xs, ys = [], []
            for s in sorted({s for (_, s) in T}, key=xval):
                if not s.startswith(c["prefix"]) or (k, s) not in T: continue
                fw = [T[(f, s)][0] for f in FW if (f, s) in T]
                if fw: xs.append(xval(s)); ys.append(T[(k, s)][0] / min(fw))
            if xs: ax.plot(xs, ys, ls, color=COL[k], marker=MRK[k], ms=8, lw=2.2, label=f"{LAB[k]} · {mode}")
    ax.axhline(1.0, color="k", lw=1.2); ax.text(c["ticks"][0] * 1.02, 1.05, "equal to best framework", fontsize=11)
    setx(ax, c); ax.set_yscale("log"); ax.set_ylabel("R = STADE / best framework"); ax.set_title(c["ratio_title"])
    if ax.lines: ax.legend(loc="best", fontsize=10.5)

def mem_value(r):
    if r["contender"] == "P-GRAPH" and r.get("M_B") is not None: return max(r["M_A"], r["M_B"])      # a CUDA graph keeps its freed buffers reserved in a private pool
    return r["M_A"]

def mem_data(recs, case):
    return {(r["contender"], r["mode"], r["size"]): r for r in recs if r.get("kind") == "mem" and r.get("case") == case and r.get("scope", "all") == "all" and r.get("M_A") is not None}

def panel_mem(ax, M, c, mode="adjoint"):
    sizes = sorted({s for (_, m, s) in M if m == mode}, key=lambda z: (0 if z.startswith(c['prefix']) else 1, xval(z)))
    if not sizes: ax.text(.5, .5, "no memory data", ha="center", transform=ax.transAxes); return
    cs = [k for k in ORDER if any((k, mode, s) in M for s in sizes)]; w = 0.8 / max(len(cs), 1); x = np.arange(len(sizes))
    for i, k in enumerate(cs):
        ax.bar(x + (i - (len(cs) - 1) / 2) * w, [mem_value(M[(k, mode, s)]) / 2**20 if (k, mode, s) in M else np.nan for s in sizes], w * 0.92, color=COL[k], label=LAB[k])
    ax.set_xticks(x); ax.set_xticklabels([c["mem_lab"](s) for s in sizes]); ax.set_yscale("log"); ax.set_ylabel("memory (MiB)"); ax.set_xlabel("number of nodes" if c["prefix"] == "nn" else "size"); ax.set_title(c["mem_title"])

MEMNOTE = "Memory: arrays + tape (STADE) · static analysis of the executable (JAX) · peak allocated (PyTorch eager) · reserved pool (PyTorch CUDA graph)"
def fig_main(recs, meta, out, case, c):
    TA, TP = times(recs, case, "adjoint"), times(recs, case, "primal"); fig, axs = plt.subplots(2, 2, figsize=(13.33, 7.5))
    fig.suptitle(f"{c['name']} on a V100: STADE-generated GPU code against PyTorch and JAX", fontsize=19, y=0.985)
    panel_time(axs[0, 0], TA, "Gradient evaluation (adjoint)", c); panel_time(axs[0, 1], TP, "Forward pass (primal)", c); panel_ratio(axs[1, 0], TP, TA, c); panel_mem(axs[1, 1], mem_data(recs, case), c)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.945), fontsize=13, frameon=False)
    footer(fig, meta, "hollow markers = wide net (32, 64)" if c["extra"] else ""); fig.text(0.01, 0.034, MEMNOTE, fontsize=10.5, color="#555"); fig.tight_layout(rect=(0, 0.06, 1, 0.9)); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".svg"); plt.close(fig)

def fig_single(recs, meta, out, case, c, which):
    TA, TP = times(recs, case, "adjoint"), times(recs, case, "primal"); fig, ax = plt.subplots(figsize=(10, 6.2))
    if which == "adjoint": panel_time(ax, TA, f"{c['name']} gradient evaluation (adjoint), batch 1", c); ax.legend(loc="upper left")
    elif which == "primal": panel_time(ax, TP, f"{c['name']} forward pass (primal), batch 1", c); ax.legend(loc="upper left")
    elif which == "ratio": panel_ratio(ax, TP, TA, c)
    elif which == "memory": panel_mem(ax, mem_data(recs, case), c); ax.legend(fontsize=11)
    footer(fig, meta, MEMNOTE if which == "memory" and False else ""); fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".svg"); plt.close(fig)

def parity(R):
    l300 = {k: R[k]["losses_at"].get("300") for k in R if R[k].get("losses_at", {}).get("300") is not None}; out = {}
    for k in l300:
        o = [v for q, v in l300.items() if q != k]; out[k] = abs(l300[k] - np.median(o)) / abs(np.median(o)) if o else 0.0
    return out

def fig_train(recs, meta, out, c):
    R = {r["contender"]: r for r in recs if r.get("kind") == "train" and r.get("model") == c["model"] and r.get("sync_step_us")}
    if not R: return False
    cs = [k for k in ORDER if k in R]; par = parity(R); bad = {k for k in cs if par.get(k, 0) > 1e-6}; x = np.arange(len(cs)); fig, axs = plt.subplots(1, 2, figsize=(13.33, 5.8))
    med = [np.median(R[k]["sync_step_us"]) / 1e3 for k in cs]; lo = [med[i] - np.percentile(R[k]["sync_step_us"], 25) / 1e3 for i, k in enumerate(cs)]; hi = [np.percentile(R[k]["sync_step_us"], 75) / 1e3 - med[i] for i, k in enumerate(cs)]
    thr = [R[k]["throughput_steps_per_s"] for k in cs]
    for ax, vals, kw in ((axs[0], med, dict(yerr=[lo, hi], capsize=4)), (axs[1], thr, {})):
        bars = ax.bar(x, vals, color=[COL[k] for k in cs], **kw)
        for b, k in zip(bars, cs):
            if k in bad: b.set_hatch("//"); b.set_edgecolor("black"); b.set_alpha(0.55)
        for i, v in enumerate(vals): ax.text(i, v, (f"{v:.0f}" if v >= 100 else f"{v:.3g}"), ha="center", va="bottom", fontsize=12, bbox=dict(fc="white", ec="none", pad=1, alpha=.7))
        ax.set_xticks(x); ax.set_xticklabels([LAB[k].replace(" ", "\n", 1) for k in cs], fontsize=11)
    axs[0].set_ylabel("time per training step (ms)"); axs[0].set_title("Step time (synchronized, median and quartiles)"); axs[1].set_ylabel("steps per second"); axs[1].set_title("Throughput (pipelined, 300 steps)")
    fig.suptitle(c["train_title"], fontsize=16, y=0.99)
    if bad: fig.text(0.5, 0.045, "Hatched = fails the parity gate: loss at step 300 differs by " + ", ".join(f"{par[k]:.1e} ({LAB[k]})" for k in cs if k in bad) + " from the other tools (limit 1e-6). Cause not investigated.", ha="center", fontsize=11.5, color="#8e1b1b")
    footer(fig, meta, "after 20 warm-up steps", what="median over 300 steps"); fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".svg"); plt.close(fig); return True

def summary_rows(recs, case):
    rows = []
    for mode in ("adjoint", "primal"):
        T = times(recs, case, mode)
        for s in sorted({s for (_, s) in T}, key=lambda z: (0 if z.startswith(CFG[case]['prefix']) else 1, xval(z))):
            fw = {k: T[(k, s)][0] for k in FW if (k, s) in T}; row = {"mode": mode, "size": s, **{k: round(T[(k, s)][0], 1) for k in ORDER if (k, s) in T}}
            if fw:
                bc = min(fw, key=fw.get); row["best_framework"] = bc
                for k in ("S-CUDA", "S-JACC"):
                    if (k, s) in T: row["R_" + k] = round(T[(k, s)][0] / fw[bc], 3)
            rows.append(row)
    return rows

if __name__ == "__main__":
    d, outd, case = sys.argv[1], sys.argv[2], sys.argv[3]; c = CFG[case]; os.makedirs(outd, exist_ok=True); tag = "unet" if case == "K7" else "mpnn"
    recs, meta = load(d); print("records used:", sum(1 for r in recs if r.get("kind") in ("time", "mem", "train")), "| jobs:", meta["jobs"], "| hosts:", sorted(meta["hosts"]), "| dropped:", dict(meta["dropped"]))
    fig_main(recs, meta, os.path.join(outd, tag + "_overview"), case, c)
    for w in ("adjoint", "primal", "ratio", "memory"): fig_single(recs, meta, os.path.join(outd, f"{tag}_{w}"), case, c, w)
    print("training figure:", fig_train(recs, meta, os.path.join(outd, tag + "_training"), c))
    rows = summary_rows(recs, case); json.dump(rows, open(os.path.join(outd, tag + "_summary.json"), "w"), indent=1)
    cols = ["mode", "size"] + ORDER + ["best_framework", "R_S-CUDA", "R_S-JACC"]
    with open(os.path.join(outd, tag + "_summary.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows: f.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in cols) + "\n")
    print("summary rows:", len(rows))
