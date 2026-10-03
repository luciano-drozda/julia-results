# Mimics the Julia input generator (same formulas) so the harness can be tested without a GPU.
import numpy as np, json, os, sys
d = sys.argv[1]; os.makedirs(d, exist_ok=True)
def f(n, a, b, c): i = np.arange(1, n + 1); return np.sin(a * i + b) + c * np.cos(0.7 * i)
nh = 64
K = {"stencil_loss": {"u": [1000]}, "stencil_loss_big": {"u": [40000]}, "matvec_loss": {"a": [30, 20], "u": [20]},
     "mlp1d": {"x": [1], "y": [1], "w1": [nh], "b1": [nh], "w2": [nh * nh], "b2": [nh], "w3": [nh], "b3": [1]}}
arrs = {}
for k, sh in K.items():
    for j, (n, s) in enumerate(sh.items()):
        sz = int(np.prod(s)); a = (0.3 * f(sz, 0.013 * (j + 1), 0.1 * j, 0.1) ).reshape(s, order="F")
        if k == "mlp1d" and n in ("w2",): a = a * 0.2
        a.flatten(order="F").astype("<f8").tofile(f"{d}/in_{k}_{n}.bin"); arrs[(k, n)] = a
json.dump({"kernels": list(K), "shapes": K, "ints": {"mlp1d": {"n_h": nh}}, "variants": []}, open(d + "/meta.json", "w"))
# analytic references
refs = {}
for kk in ("stencil_loss", "stencil_loss_big"):
    u = arrs[(kk, "u")]; w = u[:-2] - 2 * u[1:-1] + u[2:]
    g = np.zeros_like(u); g[:-2] += 2 * w; g[1:-1] -= 4 * w; g[2:] += 2 * w
    refs[kk] = (float((w ** 2).sum()), {"u": g})
a, uu = arrs[("matvec_loss", "a")], arrs[("matvec_loss", "u")]; v = a @ uu
refs["matvec_loss"] = (float((v ** 2).sum()), {"a": 2 * np.outer(v, uu), "u": 2 * a.T @ v})
M = {n: arrs[("mlp1d", n)] for n in K["mlp1d"]}; W2 = M["w2"].reshape(nh, nh)
h1 = np.tanh(M["w1"] * M["x"][0] + M["b1"]); h2 = np.tanh(W2 @ h1 + M["b2"]); o = (M["w3"] * h2).sum() + M["b3"][0]
do = 2 * (o - M["y"][0]); dh2 = do * M["w3"]; ds2 = dh2 * (1 - h2 ** 2); dh1 = W2.T @ ds2; ds1 = dh1 * (1 - h1 ** 2)
refs["mlp1d"] = (float((o - M["y"][0]) ** 2), {"x": np.array([(ds1 * M["w1"]).sum()]), "y": np.array([-do]), "w1": ds1 * M["x"][0], "b1": ds1,
                  "w2": np.outer(ds2, h1).reshape(-1), "b2": ds2, "w3": do * h2, "b3": np.array([do])})
for k, (l, gr) in refs.items():
    json.dump({"loss": l}, open(f"{d}/ref_{k}_loss.json", "w"))
    for n, x in gr.items(): np.asarray(x).flatten(order="F").astype("<f8").tofile(f"{d}/ref_{k}_{n}.bin")
print("inputs + analytic refs written to", d)
