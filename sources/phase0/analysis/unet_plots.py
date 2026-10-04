#!/usr/bin/env python3
"""Slide plots for the U-Net (K7 and training model M3) from raw result JSON files.
Usage: unet_plots.py <dir with result json files> <output dir>
Threshold-free: draws times, ratios R and memory. It draws no band or 'competitive' label."""
import sys, os, json, glob, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"S-CUDA": "#c0392b", "S-JACC": "#e67e22", "P-EAGER": "#2471a3", "P-GRAPH": "#17a589", "J-JIT": "#7d3c98"}
LAB = {"S-CUDA": "STADE (CUDA)", "S-JACC": "STADE (JACC)", "P-EAGER": "PyTorch eager", "P-GRAPH": "PyTorch CUDA graph", "J-JIT": "JAX jit"}
MRK = {"S-CUDA": "o", "S-JACC": "s", "P-EAGER": "^", "P-GRAPH": "v", "J-JIT": "D"}
ORDER = ["S-CUDA", "S-JACC", "P-EAGER", "P-GRAPH", "J-JIT"]
FW = ("P-EAGER", "P-GRAPH", "J-JIT")
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "legend.fontsize": 13, "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.grid": True, "grid.alpha": .3})

def load(d):
    """Timing and training records are kept only from passes that recorded no foreign GPU process (field `cotenant`).
    Memory records are exact byte counts per process, so they are kept from any pass."""
    recs, meta = [], {"dropped": collections.Counter(), "jobs": [], "hosts": set()}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try: j = json.load(open(p))
        except Exception: continue
        if not isinstance(j, dict) or "records" not in j: continue
        meta["jobs"].append(j.get("job")); meta["hosts"].add(j.get("host"))
        meta["gpu"] = (j.get("snap_start", {}).get("gpu", "") or "").split(",")[0] or meta.get("gpu")
        ok = {}
        for i, ps in enumerate(j.get("passes", []), start=1):
            c = ps.get("cotenant"); ok[i] = (c is not None and c.get("foreign_processes_after_wait", 1) == 0)
        for r in j["records"]:
            if r.get("kind") in ("time", "train") and not ok.get(r.get("pass"), False):
                meta["dropped"][(j.get("job"), r.get("kind"))] += 1; continue
            r["_job"] = j.get("job"); recs.append(r)
    return recs, meta

def size_side(sid): return int(sid.split("-")[1])       # narrow-64 -> 64

def times(recs, mode):
    """(contender, sizeid) -> (median, p25, p75) in microseconds. Only scope 'all', K7."""
    out = {}
    for r in recs:
        if r.get("kind") == "time" and r.get("case") == "K7" and r.get("mode") == mode and r.get("scope", "all") == "all" and r.get("trial_us"):
            v = np.asarray(r["trial_us"], float); out[(r["contender"], r["size"])] = (np.median(v), np.percentile(v, 25), np.percentile(v, 75))
    return out

def footer(fig, meta, extra=""):
    gpu = meta.get("gpu") or "GPU"
    fig.text(0.01, 0.008, f"{gpu} · float64 · batch 1 · median of 30 trials, bars = quartiles · ONE repeat, no stability check{(' · ' + extra) if extra else ''}", fontsize=10.5, color="#555")

def panel_time(ax, T, title, kind):
    for c in ORDER:
        for fam, ls, fill in (("narrow", "-", True), ("wide", "", False)):
            pts = sorted([(size_side(s), v) for (cc, s), v in T.items() if cc == c and s.startswith(fam)])
            if not pts: continue
            x = [p[0] for p in pts]; y = [p[1][0] for p in pts]; lo = [p[1][0] - p[1][1] for p in pts]; hi = [p[1][2] - p[1][0] for p in pts]
            ax.errorbar(x, y, yerr=[lo, hi], color=COL[c], marker=MRK[c], ms=8, lw=2.2 if ls else 0, ls=ls or "none", mfc=COL[c] if fill else "white", mec=COL[c], mew=2, capsize=3,
                        label=LAB[c] if fam == "narrow" or not any(cc == c and s.startswith("narrow") for (cc, s) in T) else None)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([16, 32, 64, 128, 256]); ax.set_xticklabels(["16", "32", "64", "128", "256"])
    ax.set_xlabel("image side h = w (pixels)"); ax.set_ylabel("time per call (µs)"); ax.set_title(title)

def panel_ratio(ax, TP, TA):
    for mode, T, ls in (("adjoint", TA, "-"), ("primal", TP, "--")):
        for c in ("S-CUDA", "S-JACC"):
            xs, ys = [], []
            for s in sorted({s for (cc, s) in T}, key=lambda s: (s.split("-")[0], size_side(s))):
                if not s.startswith("narrow") or (c, s) not in T: continue
                fw = [T[(f, s)][0] for f in FW if (f, s) in T]
                if fw: xs.append(size_side(s)); ys.append(T[(c, s)][0] / min(fw))
            if xs: ax.plot(xs, ys, ls, color=COL[c], marker=MRK[c], ms=8, lw=2.2, label=f"{LAB[c]} · {mode}")
    ax.axhline(1.0, color="k", lw=1.2); ax.text(16.5, 1.05, "equal to best framework", fontsize=11)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([16, 32, 64, 128, 256]); ax.set_xticklabels(["16", "32", "64", "128", "256"])
    ax.set_xlabel("image side h = w (pixels)"); ax.set_ylabel("R = STADE / best framework"); ax.set_title("Ratio to the best framework (narrow net)")
    if ax.lines: ax.legend(loc="lower right", fontsize=10.5)

def mem_data(recs):
    M = {}
    for r in recs:
        if r.get("kind") == "mem" and r.get("case") == "K7" and r.get("scope", "all") == "all" and r.get("M_A") is not None:
            M[(r["contender"], r["mode"], r["size"])] = r
    return M

def mem_value(r):
    """Footprint in bytes. A CUDA graph replays inside a private pool: freed activation buffers stay reserved, so use the reserved pool for it."""
    if r["contender"] == "P-GRAPH" and r.get("M_B") is not None: return max(r["M_A"], r["M_B"])
    return r["M_A"]

def panel_mem(ax, M, mode="adjoint"):
    sizes = sorted({s for (_, m, s) in M if m == mode}, key=lambda s: (s.split("-")[0], size_side(s)))
    if not sizes: ax.text(.5, .5, "no memory data", ha="center", transform=ax.transAxes); return
    cs = [c for c in ORDER if any((c, mode, s) in M for s in sizes)]
    w = 0.8 / max(len(cs), 1); x = np.arange(len(sizes))
    for i, c in enumerate(cs):
        ys = [mem_value(M[(c, mode, s)]) / 2**20 if (c, mode, s) in M else np.nan for s in sizes]
        ax.bar(x + (i - (len(cs) - 1) / 2) * w, ys, w * 0.92, color=COL[c], label=LAB[c])
    ax.set_xticks(x); ax.set_xticklabels([s.replace("narrow-", "n-").replace("wide-", "w-") for s in sizes]); ax.set_yscale("log")
    ax.set_ylabel("memory (MiB)"); ax.set_xlabel("U-Net size"); ax.set_title("Memory (arrays + tape + temporaries)")

def fig_main(recs, meta, out):
    TA, TP = times(recs, "adjoint"), times(recs, "primal"); M = mem_data(recs)
    fig, axs = plt.subplots(2, 2, figsize=(13.33, 7.5)); fig.suptitle("U-Net on a V100: STADE-generated GPU code against PyTorch and JAX", fontsize=19, y=0.985)
    panel_time(axs[0, 0], TA, "Gradient evaluation (adjoint)", "adjoint"); panel_time(axs[0, 1], TP, "Forward pass (primal)", "primal")
    panel_ratio(axs[1, 0], TP, TA); panel_mem(axs[1, 1], M, "adjoint")
    h, l = axs[0, 0].get_legend_handles_labels()
    if not h: h, l = axs[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.945), fontsize=13, frameon=False)
    footer(fig, meta, "hollow markers = wide net (32, 64)"); fig.text(0.01, 0.034, "Memory: arrays + tape (STADE) · static analysis of the executable (JAX) · peak allocated (PyTorch eager) · reserved pool (PyTorch CUDA graph)", fontsize=10.5, color="#555"); fig.tight_layout(rect=(0, 0.06, 1, 0.9)); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".svg"); plt.close(fig)

def fig_single(recs, meta, out, which):
    TA, TP = times(recs, "adjoint"), times(recs, "primal"); fig, ax = plt.subplots(figsize=(10, 6.2))
    if which == "adjoint": panel_time(ax, TA, "U-Net gradient evaluation (adjoint), batch 1", "adjoint"); ax.legend(loc="upper left")
    elif which == "primal": panel_time(ax, TP, "U-Net forward pass (primal), batch 1", "primal"); ax.legend(loc="upper left")
    elif which == "ratio": panel_ratio(ax, TP, TA)
    elif which == "memory": panel_mem(ax, mem_data(recs)); ax.legend(fontsize=11)
    footer(fig, meta); fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".svg"); plt.close(fig)

def parity(R):
    """Relative difference of the step-300 loss against the median of the other tools. The plan's gate is 1e-6."""
    l300 = {c: R[c]["losses_at"].get("300") for c in R if R[c].get("losses_at", {}).get("300") is not None}
    out = {}
    for c in l300:
        others = [v for k, v in l300.items() if k != c]
        out[c] = abs(l300[c] - np.median(others)) / abs(np.median(others)) if others else 0.0
    return out

def fig_train(recs, meta, out):
    R = {r["contender"]: r for r in recs if r.get("kind") == "train" and r.get("model") == "M3" and r.get("sync_step_us")}
    if not R: return False
    cs = [c for c in ORDER if c in R]; par = parity(R); bad = {c for c in cs if par.get(c, 0) > 1e-6}
    fig, axs = plt.subplots(1, 2, figsize=(13.33, 5.8)); x = np.arange(len(cs))
    med = [np.median(R[c]["sync_step_us"]) / 1e3 for c in cs]; lo = [med[i] - np.percentile(R[c]["sync_step_us"], 25) / 1e3 for i, c in enumerate(cs)]; hi = [np.percentile(R[c]["sync_step_us"], 75) / 1e3 - med[i] for i, c in enumerate(cs)]
    thr = [R[c]["throughput_steps_per_s"] for c in cs]
    for ax, vals, kw in ((axs[0], med, dict(yerr=[lo, hi], capsize=4)), (axs[1], thr, {})):
        bars = ax.bar(x, vals, color=[COL[c] for c in cs], **kw)
        for b, c in zip(bars, cs):
            if c in bad: b.set_hatch("//"); b.set_edgecolor("black"); b.set_alpha(0.55)
        for i, v in enumerate(vals): ax.text(i, v, (f"{v:.0f}" if v >= 100 else f"{v:.3g}"), ha="center", va="bottom", fontsize=12, bbox=dict(fc="white", ec="none", pad=1, alpha=.7))
        ax.set_xticks(x); ax.set_xticklabels([LAB[c].replace(" ", "\n", 1) for c in cs], fontsize=11)
    axs[0].set_ylabel("time per training step (ms)"); axs[0].set_title("Step time (synchronized, median and quartiles)"); axs[1].set_ylabel("steps per second"); axs[1].set_title("Throughput (pipelined, 300 steps)")
    fig.suptitle("U-Net training, batch size 1, plain SGD (32×32 input, 3→8→16→32 channels)", fontsize=17, y=0.99)
    note = ""
    if bad: note = "Hatched = fails the parity gate: loss at step 300 differs by " + ", ".join(f"{par[c]:.1e} ({LAB[c]})" for c in cs if c in bad) + " from the other tools (limit 1e-6). Cause not investigated."
    if note: fig.text(0.5, 0.045, note, ha="center", fontsize=11.5, color="#8e1b1b")
    footer(fig, meta, "300 measured steps after 20 warm-up"); fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".svg"); plt.close(fig); return True

def summary_table(recs):
    TA, TP = times(recs, "adjoint"), times(recs, "primal"); rows = []
    for mode, T in (("adjoint", TA), ("primal", TP)):
        for s in sorted({s for (_, s) in T}, key=lambda s: (s.split("-")[0], size_side(s))):
            fw = {c: T[(c, s)][0] for c in FW if (c, s) in T}
            row = {"mode": mode, "size": s, **{c: round(T[(c, s)][0], 1) for c in ORDER if (c, s) in T}}
            if fw:
                bc = min(fw, key=fw.get); row["best_framework"] = bc
                for c in ("S-CUDA", "S-JACC"):
                    if (c, s) in T: row["R_" + c] = round(T[(c, s)][0] / fw[bc], 3)
            rows.append(row)
    return rows

if __name__ == "__main__":
    d, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    recs, meta = load(d); n = sum(1 for r in recs if r.get("kind") in ("time", "mem", "train"))
    print("records used:", n, "| jobs:", meta.get("jobs"), "| hosts:", sorted(meta.get("hosts", [])), "| dropped (timing/training from passes without a clean-GPU record):", dict(meta["dropped"]))
    fig_main(recs, meta, os.path.join(out, "unet_overview"))
    for w in ("adjoint", "primal", "ratio", "memory"): fig_single(recs, meta, os.path.join(out, "unet_" + w), w)
    print("training figure:", fig_train(recs, meta, os.path.join(out, "unet_training")))
    rows = summary_table(recs); json.dump(rows, open(os.path.join(out, "unet_summary.json"), "w"), indent=1)
    cols = ["mode", "size"] + ORDER + ["best_framework", "R_S-CUDA", "R_S-JACC"]
    with open(os.path.join(out, "unet_summary.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows: f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
    print("rows in summary:", len(rows))
