import json, os, sys, shutil, tempfile, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

def trials(m, rel=0.01, seed=0):
    rng = np.random.default_rng(seed); return list(m * (1 + rel * rng.standard_normal(30)))
def trec(case, size, mode, c, m, rel=0.01, scope="all", tool="X", seed=0):
    return dict(kind="time", case=case, size=size, mode=mode, contender=c, scope=scope, trial_us=trials(m, rel, seed), K=10, tool=tool)
def job(name, recs, host="n1"): return dict(job=name, host=host, records=recs)

tmp = tempfile.mkdtemp(); raw = os.path.join(tmp, "raw"); out = os.path.join(tmp, "out")
def put(name, d):
    p = os.path.join(raw, "p2", name); os.makedirs(p); json.dump(d, open(os.path.join(p, "result.json"), "w"))
for j, (hostn, f) in enumerate((("n1", 1.0), ("n2", 1.02))):
    recs = []
    # K1 adjoint: best framework = P-GRAPH 80. S-CUDA 70 -> R 0.875 (faster); S-JACC 100 -> R 1.25 (competitive)
    for c, m in (("P-EAGER", 100), ("P-GRAPH", 80), ("J-JIT", 120), ("S-CUDA", 70), ("S-JACC", 100)): recs.append(trec("K1", "n", "adjoint", c, m * f, seed=j))
    # a params-only row must NOT be used as best framework even if faster
    recs.append(trec("K1", "n", "adjoint", "P-EAGER(params)", 10 * f, scope="params", seed=j))
    # K3 adjoint: frameworks 10,12,15 -> best 10; S-CUDA 35 -> R 3.5 (slower); S-JACC 200 -> R 20 (far slower)
    for c, m in (("P-EAGER", 10), ("P-GRAPH", 12), ("J-JIT", 15), ("S-CUDA", 35), ("S-JACC", 200)): recs.append(trec("K3", "n", "adjoint", c, m * f, seed=j))
    # K2 primal: P-GRAPH 5; S-CUDA 8 -> R 1.6 (competitive); unstable S-JACC: 30% spread between jobs
    for c, m in (("P-EAGER", 9), ("P-GRAPH", 5), ("J-JIT", 7), ("S-CUDA", 8)): recs.append(trec("K2", "n", "primal", c, m * f, seed=j))
    recs.append(trec("K2", "n", "primal", "S-JACC", 12 * (1.0 if j == 0 else 1.6), seed=j))
    recs.append(dict(kind="gate", case="K1", size="n", mode="adjoint", contender="S-CUDA", max_err=1e-15, passed=True, determinism_err=1e-16))
    recs.append(dict(kind="mem", case="K1", size="n", mode="adjoint", contender="S-CUDA", M_A=2000, M_B=2500, M_C=2100))
    recs.append(dict(kind="mem", case="K1", size="n", mode="adjoint", contender="P-EAGER", M_A=1000, M_B=1500, M_C=1100))
    recs.append(dict(kind="mem", case="K1", size="n", mode="adjoint", contender="J-JIT", M_A=1800, M_B=1900, M_C=1850))
    recs.append(dict(kind="train", model="M3", size="s", contender="S-CUDA", sync_step_us=[100.0] * 300, throughput_steps_per_s=9000.0, losses_at={"1": 1.0, "300": 0.5}))
    put(f"jobA{j}", job(f"jobA{j}", recs, hostn))

# an excluded case (K5) and a preflight job must be ignored by the comparison
put("jobPre", job("bench-j0", [trec("K1", "n", "adjoint", c, m, seed=9) for c, m in (("P-EAGER", 1), ("S-CUDA", 1000))]))
put("jobExcl", job("jobExcl", [trec("K5", "n", "adjoint", c, m, seed=7) for c, m in (("P-EAGER", 10), ("S-CUDA", 5000))] + [dict(kind="train", model="M1", size="s", contender="S-CUDA", sync_step_us=[1.0] * 300, throughput_steps_per_s=1.0, losses_at={"1": 1.0})]))
s = A.report(raw, out)
assert s["excluded_records_dropped"] >= 3, s["excluded_records_dropped"]
assert not any(r["case"] == "K5" for r in A.ratios(A.time_table(A.collect(A.load_results(raw))[0]))), "an excluded case produced a ratio"
assert "K5" in open(os.path.join(out, "excluded_cases.csv")).read()
trc = open(os.path.join(out, "training_summary.csv")).read(); assert "M1" not in trc and "M3" in trc, trc     # the excluded training model is dropped, the included one stays
R = {(r["case"], r["mode"], r["stade"]): r for r in json.load(open(os.path.join(out, "performance_profile.json"))) and A.ratios(A.time_table(A.collect(A.load_results(raw))[0]))}
def close(a, b, tol=0.05): return abs(a - b) / b < tol
assert close(R[("K1", "adjoint", "S-CUDA")]["R"], 70 / 80), R[("K1", "adjoint", "S-CUDA")]
assert R[("K1", "adjoint", "S-CUDA")]["best_framework"] == "P-GRAPH"        # the faster params-only row is excluded
assert close(R[("K1", "adjoint", "S-JACC")]["R"], 100 / 80); assert close(R[("K3", "adjoint", "S-CUDA")]["R"], 3.5); assert close(R[("K3", "adjoint", "S-JACC")]["R"], 20.0)
assert close(R[("K2", "primal", "S-CUDA")]["R"], 8 / 5); assert close(R[("K1", "adjoint", "S-CUDA")]["R_vs_P-EAGER"], 0.7); assert close(R[("K1", "adjoint", "S-CUDA")]["R_vs_J-JIT"], 70 / 120)
assert s["bands"].startswith("NOT COMPUTED"), "band counts must be blocked before thresholds are frozen"
assert not os.path.exists(os.path.join(out, "ratios_with_bands.csv"))
unst = open(os.path.join(out, "unstable_cases.csv")).read(); assert "S-JACC" in unst and unst.count("\n") == 2, unst      # only K2 S-JACC is flagged
# freeze, then bands appear and match the hand calculation
A.freeze(out, [1.0, 2.0, 5.0], [1.0, 1.5]); s = A.report(raw, out)
b = s["bands"]; assert b["adjoint/S-CUDA"] == {"faster": 1, "slower": 1}, b
assert b["adjoint/S-JACC"] == {"competitive": 1, "far slower": 1}, b; assert b["primal/S-CUDA"] == {"competitive": 1}, b
assert s["memory_bands"] == {"larger": 1}, s["memory_bands"]                                                                  # 2000 / 1000 = 2.0 > 1.5 vs the best framework
assert s["sensitivity"]["1.5/3.0"]["adjoint/S-CUDA"] == {"faster": 1, "far slower": 1}, s["sensitivity"]   # 3.5 > 3.0
assert s["sensitivity"]["3.0/10.0"]["adjoint/S-CUDA"] == {"faster": 1, "slower": 1}, s["sensitivity"]   # 3.5 is slower than 3.0 and not above 10.0
assert os.path.exists(os.path.join(out, "performance_profile.png")), "plot missing"
prof = json.load(open(os.path.join(out, "performance_profile.json")))["adjoint/S-CUDA"]; assert prof[-1][1] == 1.0 and prof[0][1] <= 0.5
geo = s["geomean_R"]["adjoint/S-CUDA"]; assert close(geo, math.sqrt(70 / 80 * 3.5)), geo
# tampering with the frozen thresholds must be detected
th = json.load(open(os.path.join(out, "thresholds.json"))); th["runtime"]["competitive"] = 3.0; json.dump(th, open(os.path.join(out, "thresholds.json"), "w"), indent=1)
try: A.report(raw, out); raise AssertionError("tampering was not detected")
except SystemExit as e: assert "does not match" in str(e)
tr = open(os.path.join(out, "training_summary.csv")).read(); assert "M3" in tr and "9000" in tr
print("analysis tests passed:", {"ratios checked": 8, "bands": b["adjoint/S-CUDA"], "geomean_R": round(geo, 3), "unstable flagged": 1, "tamper detected": True})
shutil.rmtree(tmp)
