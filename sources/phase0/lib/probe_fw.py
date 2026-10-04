"""Capability probes for job J0. Usage: python probe_fw.py torch|jax out.json"""
import sys, json, time, traceback, importlib.metadata as md
fw, outp = sys.argv[1], sys.argv[2]
out = {"fw": fw}
def step(name, fn):
    t0 = time.perf_counter()
    try: out[name] = fn(); 
    except Exception as e: out[name] = {"ok": False, "error": repr(e)[:400], "trace": traceback.format_exc()[-600:]}
    if isinstance(out[name], dict): out[name]["seconds"] = time.perf_counter() - t0
def pkgs(names):
    r = {}
    for p in names:
        try: r[p] = md.version(p)
        except Exception: r[p] = None
    return r
import numpy as np
if fw == "torch":
    import torch, torch.nn.functional as F
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out["versions"] = dict(torch=torch.__version__, cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(), available=torch.cuda.is_available(),
                           device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, packages=pkgs(["torch", "numpy", "triton", "nvidia-cudnn-cu12", "nvidia-cublas-cu12"]))
    dev = "cuda"
    def conv():
        x = torch.randn(1, 4, 16, 16, dtype=torch.float64, device=dev); w = torch.randn(6, 4, 3, 3, dtype=torch.float64, device=dev, requires_grad=True)
        y = F.conv2d(x, w, padding=1); (y ** 2).sum().backward()
        ref = F.conv2d(x.cpu(), w.detach().cpu(), padding=1)
        return {"ok": True, "fwd_err": float((y.cpu() - ref).abs().max()), "grad_finite": bool(torch.isfinite(w.grad).all())}
    def ops():
        idx = torch.randint(0, 5, (20,), device=dev); v = torch.randn(20, 3, dtype=torch.float64, device=dev)
        a = torch.zeros(5, 3, dtype=torch.float64, device=dev).index_add(0, idx, v)
        ref = torch.zeros(5, 3, dtype=torch.float64); ref.index_add_(0, idx.cpu(), v.cpu())
        p = F.max_pool2d(torch.randn(1, 2, 8, 8, dtype=torch.float64, device=dev), 2); u = p.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        return {"ok": True, "index_add_err": float((a.cpu() - ref).abs().max()), "upsample_shape": list(u.shape)}
    def graph():
        W = torch.randn(64, 64, dtype=torch.float64, device=dev, requires_grad=True); x = torch.randn(64, dtype=torch.float64, device=dev)
        def body():
            if W.grad is not None: W.grad.zero_()
            l = (torch.tanh(W @ x) ** 2).sum(); l.backward(); return l.detach()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): holder = body()
        x.copy_(torch.randn(64, dtype=torch.float64, device=dev)); g.replay(); torch.cuda.synchronize()
        gr = W.grad.clone(); W.grad.zero_(); l2 = (torch.tanh(W @ x) ** 2).sum(); l2.backward()
        return {"ok": True, "replay_vs_eager_err": float((gr - W.grad).abs().max()), "loss_err": float(abs(holder - l2.detach()))}
    def comp():
        f = torch.compile(lambda a, b: (torch.tanh(a @ b) ** 2).sum())
        a = torch.randn(64, 64, dtype=torch.float64, device=dev, requires_grad=True); b = torch.randn(64, 64, dtype=torch.float64, device=dev)
        t0 = time.perf_counter(); l = f(a, b); l.backward(); torch.cuda.synchronize(); t1 = time.perf_counter() - t0
        ref = (torch.tanh(a @ b) ** 2).sum()
        return {"ok": True, "compile_and_first_call_s": t1, "loss_err": float(abs(l.detach() - ref.detach()))}
    def mem():
        free, total = torch.cuda.mem_get_info(); torch.cuda.reset_peak_memory_stats(); x = torch.zeros(10_000_000, dtype=torch.float64, device=dev)
        return {"ok": True, "free_MiB": free / 2**20, "total_MiB": total / 2**20, "peak_after_80MB_MiB": torch.cuda.max_memory_allocated() / 2**20}
    for nm, fn in (("conv_f64", conv), ("ops", ops), ("cuda_graph", graph), ("torch_compile", comp), ("memory_apis", mem)): step(nm, fn)
else:
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax, jax.numpy as jnp
    from jax import lax
    jax.config.update("jax_enable_x64", True)
    d0 = jax.devices()[0]
    out["versions"] = dict(jax=jax.__version__, backend=jax.default_backend(), device=str(d0), on_gpu=jax.default_backend() in ("gpu", "cuda"),
                           packages=pkgs(["jax", "jaxlib", "jax-cuda12-plugin", "nvidia-cudnn-cu12", "flax", "optax"]),
                           remat_pass=str(getattr(jax.config, "jax_compiler_enable_remat_pass", "n/a")), xla_flags=os.environ.get("XLA_FLAGS"))
    DN = ("NCHW", "OIHW", "NCHW")
    def conv():
        rng = np.random.default_rng(1); x = rng.standard_normal((1, 4, 16, 16)); w = rng.standard_normal((6, 4, 3, 3))
        f = lambda w_: jnp.sum(lax.conv_general_dilated(jnp.asarray(x), w_, (1, 1), "SAME", dimension_numbers=DN) ** 2)
        g = jax.jit(jax.grad(f))(jnp.asarray(w)); g.block_until_ready()
        e = 1e-6; w2 = w.copy(); w2[1, 2, 0, 1] += e; w3 = w.copy(); w3[1, 2, 0, 1] -= e
        fd = (float(f(jnp.asarray(w2))) - float(f(jnp.asarray(w3)))) / (2 * e)
        return {"ok": True, "grad_finite": bool(jnp.all(jnp.isfinite(g))), "fd_rel_err": abs(fd - float(g[1, 2, 0, 1])) / abs(fd)}
    def opsf():
        rng = np.random.default_rng(2); idx = rng.integers(0, 5, 20); v = rng.standard_normal((20, 3))
        a = jnp.zeros((5, 3)).at[jnp.asarray(idx)].add(jnp.asarray(v)); ref = np.zeros((5, 3)); np.add.at(ref, idx, v)
        p = lax.reduce_window(jnp.asarray(rng.standard_normal((1, 2, 8, 8))), -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID")
        u = jnp.repeat(jnp.repeat(p, 2, axis=2), 2, axis=3)
        return {"ok": True, "scatter_err": float(np.abs(np.asarray(a) - ref).max()), "upsample_shape": list(u.shape), "softmax_ok": bool(jnp.allclose(jax.nn.softmax(jnp.arange(4.0)).sum(), 1.0))}
    def scan():
        def loss(u0):
            u, _ = lax.scan(lambda u, _: (u.at[1:].set(u[1:] - 0.1 * (u[1:] - u[:-1])), None), u0, None, length=50); return jnp.sum(u ** 2)
        g = jax.jit(jax.grad(loss))(jnp.linspace(0, 1, 100)); return {"ok": True, "finite": bool(jnp.all(jnp.isfinite(g)))}
    def memana():
        f = jax.jit(lambda a, b: (a @ b).sum()); a = jnp.ones((256, 256)); ma = f.lower(a, a).compile().memory_analysis()
        s0 = d0.memory_stats(); big = jnp.ones((10_000_000,)).block_until_ready(); s1 = d0.memory_stats()
        return {"ok": True, "memory_analysis": {k: getattr(ma, k, None) for k in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")},
                "peak_before": s0.get("peak_bytes_in_use"), "peak_after_80MB": s1.get("peak_bytes_in_use"), "has_reset_method": hasattr(d0, "reset_peak_memory_stats")}
    def ckpt():
        f = lambda x: jnp.sum(jnp.tanh(x @ x.T) ** 2); x = jnp.asarray(np.random.default_rng(3).standard_normal((16, 16)))
        g1 = jax.grad(f)(x); g2 = jax.grad(jax.checkpoint(f))(x)
        return {"ok": True, "checkpoint_err": float(jnp.abs(g1 - g2).max())}
    def flaxo():
        from flax import nnx; import optax
        lin = nnx.Linear(8, 4, rngs=nnx.Rngs(0), dtype=jnp.float64, param_dtype=jnp.float64); y = lin(jnp.ones((2, 8), jnp.float64))
        opt = optax.sgd(0.1); st = opt.init({"w": jnp.ones(3)}); return {"ok": True, "dtype": str(y.dtype)}
    def convspeed():
        x = jnp.asarray(np.random.default_rng(4).standard_normal((1, 8, 64, 64))); w = jnp.asarray(np.random.default_rng(5).standard_normal((8, 8, 3, 3)))
        f = jax.jit(lambda a, b: lax.conv_general_dilated(a, b, (1, 1), "SAME", dimension_numbers=DN)); f(x, w).block_until_ready()
        t0 = time.perf_counter(); [f(x, w) for _ in range(10)][-1].block_until_ready(); return {"ok": True, "ten_calls_ms": (time.perf_counter() - t0) * 1e3}
    for nm, fn in (("conv_f64", conv), ("ops", opsf), ("scan_grad", scan), ("memory_tools", memana), ("checkpoint", ckpt), ("flax_optax", flaxo), ("conv_speed", convspeed)): step(nm, fn)
json.dump(out, open(outp, "w")); print("probe done")
