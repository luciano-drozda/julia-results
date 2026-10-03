# Single-precision health check harness. Modes: torch | jax64 | jax32
# Reads inputs written by Julia, computes gradients, compares with STADE outputs and a float64 reference.
import sys, os, json, traceback
import numpy as np
mode, d = sys.argv[1], sys.argv[2]
meta = json.load(open(d + "/meta.json"))
DIFF = {"stencil_loss": ["u"], "matvec_loss": ["a", "u"],
        "mlp1d": ["x", "y", "w1", "b1", "w2", "b2", "w3", "b3"]}
DIFF["stencil_loss_big"] = DIFF["stencil_loss"]
def base(k): return k.replace("_big", "")
def load(path, shape):
    return np.fromfile(path, dtype="<f8").reshape(shape, order="F")
def rel(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
def inputs(k):
    return {n: load(f"{d}/in_{k}_{n}.bin", s) for n, s in meta["shapes"][k].items()}
def errs(k, loss, grads, ref_loss, ref_grads):
    e = {n: rel(grads[n], ref_grads[n]) for n in DIFF[k]}
    e_loss = abs(loss - ref_loss) / max(abs(ref_loss), 1e-300)
    return {"loss_rel_err": float(e_loss), "grad_rel_err": e, "max_rel_err": float(max([e_loss] + list(e.values())))}

out = {"mode": mode}
if mode == "torch":
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out["torch"] = torch.__version__; out["device"] = torch.cuda.get_device_name(0)
    def loss_t(k, T):
        k = base(k)
        if k == "stencil_loss":
            u = T["u"]; w = u[:-2] - 2.0 * u[1:-1] + u[2:]; return (w * w).sum()
        if k == "matvec_loss":
            v = T["a"] @ T["u"]; return (v * v).sum()
        nh = meta["ints"]["mlp1d"]["n_h"]
        h1 = torch.tanh(T["w1"] * T["x"][0] + T["b1"])
        h2 = torch.tanh(T["w2"].reshape(nh, nh) @ h1 + T["b2"])
        o = (T["w3"] * h2).sum() + T["b3"][0]
        return (o - T["y"][0]) ** 2
    def run(k, dt):
        I = inputs(k)
        T = {n: torch.tensor(a, dtype=dt, device="cuda", requires_grad=(n in DIFF[k])) for n, a in I.items()}
        L = loss_t(k, T); L.backward(); torch.cuda.synchronize()
        return float(L), {n: T[n].grad.double().cpu().numpy() for n in DIFF[k]}, str(T[DIFF[k][0]].grad.dtype)
    for k in meta["kernels"]:
        try:
            ref_l, ref_g, _ = run(k, torch.float64)
            for n, g in ref_g.items(): np.asarray(g).flatten(order="F").astype("<f8").tofile(f"{d}/ref_{k}_{n}.bin")  # column-major, like Julia
            json.dump({"loss": ref_l}, open(f"{d}/ref_{k}_loss.json", "w"))
            r = {"torch_f64": dict(errs(k, ref_l, ref_g, ref_l, ref_g), note="reference")}
            l32, g32, dt32 = run(k, torch.float32)
            r["torch_f32"] = dict(errs(k, l32, g32, ref_l, ref_g), grad_dtype=dt32)
            for var in meta["variants"]:
                lp = f"{d}/out_{k}_{var}_loss.bin"
                if not os.path.exists(lp): continue
                sl = float(np.fromfile(lp, dtype="<f8")[0])
                sg = {n: load(f"{d}/out_{k}_{var}_{n}.bin", ref_g[n].shape) for n in DIFF[k]}
                r["stade_" + var] = errs(k, sl, sg, ref_l, ref_g)
            out[k] = r
        except Exception as ex:
            out[k] = {"error": repr(ex)[:400], "trace": traceback.format_exc()[-700:]}
else:
    import jax
    if mode == "jax64": jax.config.update("jax_enable_x64", True)
    else: jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    out["jax"] = jax.__version__; out["backend"] = jax.default_backend(); out["x64"] = bool(jax.config.jax_enable_x64)
    if out["backend"] not in ("gpu", "cuda") and os.environ.get("SP_ALLOW_CPU") != "1":
        raise RuntimeError("JAX is not on the GPU: " + out["backend"])
    dt = jnp.float64 if mode == "jax64" else jnp.float32
    def loss_j(k, T):
        k = base(k)
        if k == "stencil_loss":
            u = T["u"]; w = u[:-2] - 2.0 * u[1:-1] + u[2:]; return jnp.sum(w * w)
        if k == "matvec_loss":
            v = T["a"] @ T["u"]; return jnp.sum(v * v)
        nh = meta["ints"]["mlp1d"]["n_h"]
        h1 = jnp.tanh(T["w1"] * T["x"][0] + T["b1"])
        h2 = jnp.tanh(T["w2"].reshape(nh, nh) @ h1 + T["b2"])
        o = jnp.sum(T["w3"] * h2) + T["b3"][0]
        return (o - T["y"][0]) ** 2
    for k in meta["kernels"]:
        try:
            I = {n: jnp.asarray(a, dtype=dt) for n, a in inputs(k).items()}
            diff = {n: I[n] for n in DIFF[k]}; rest = {n: v for n, v in I.items() if n not in diff}
            f = jax.jit(jax.value_and_grad(lambda D: loss_j(k, {**D, **rest})))
            L, G = f(diff); L.block_until_ready()
            ref_l = json.load(open(f"{d}/ref_{k}_loss.json"))["loss"]
            ref_g = {n: load(f"{d}/ref_{k}_{n}.bin", np.asarray(diff[n]).shape) for n in DIFF[k]}
            out[k] = {"jax_" + mode[3:]: dict(errs(k, float(L), {n: np.asarray(G[n]) for n in DIFF[k]}, ref_l, ref_g),
                                              grad_dtype=str(G[DIFF[k][0]].dtype))}
        except Exception as ex:
            out[k] = {"error": repr(ex)[:400], "trace": traceback.format_exc()[-700:]}
print(json.dumps(out))
