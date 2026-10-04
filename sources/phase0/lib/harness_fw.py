"""Benchmark harness for PyTorch and JAX. One process per launch.
Usage: python harness_fw.py --fw torch|jax --tasks tasks.json --out out.json [--ref ref.json] [--write-ref]
Prints one JSON line (the same content as --out)."""
import sys, os, json, time, gc, math, argparse, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as L
import bench_models as M

ap = argparse.ArgumentParser()
ap.add_argument("--fw", required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ref", default=None); ap.add_argument("--write-ref", action="store_true"); ap.add_argument("--spec", default=None)
args = ap.parse_args()
T0 = time.perf_counter()
TASKS = json.load(open(args.tasks))
L.load_spec(args.spec or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases_spec.json"))
FW = args.fw
ALLOW_CPU = os.environ.get("BENCH_ALLOW_CPU") == "1"

if FW == "torch":
    import torch
    ops = M.TorchOps()
    DEV = "cuda"
    if not torch.cuda.is_available() and not ALLOW_CPU: raise RuntimeError("torch.cuda.is_available() is False")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = bool(TASKS.get("cudnn_benchmark", True))
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    def sync(): torch.cuda.synchronize() if DEV == "cuda" else None
    def to_dev(a): return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
    def to_np(t): return t.detach().cpu().numpy()
else:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    ops = M.JaxOps()
    if jax.default_backend() not in ("gpu", "cuda") and not ALLOW_CPU: raise RuntimeError("JAX is not on the GPU: " + jax.default_backend())
    def sync(): pass
    def to_dev(a): return jnp.asarray(np.ascontiguousarray(a))
    def to_np(t): return np.asarray(t)

TIME = TASKS.get("time", {})
GATE_TOL = TASKS.get("gate_tol", 1e-9)
DEADLINE = TASKS.get("deadline_unix")
REF = json.load(open(args.ref)) if (args.ref and os.path.exists(args.ref) and not args.write_ref) else {}
NEWREF = {}
OUT = {"fw": FW, "records": [], "env": {}, "errors": []}

def log(msg): print(msg, file=sys.stderr, flush=True)

# ---------------------------------------------------------------- building
def build(fam, params, mode):
    """Framework arrays for one size. mode 'adjoint' or 'primal'. Returns (P, c, host_arrays)."""
    arrays, scalars, ints = L.make_data(fam, params)
    c = dict(L.fam_env(fam, params)); c.update(ints)
    if "eps" in scalars: c["eps"] = scalars["eps"]
    P = {}
    F = L.SPEC["families"][fam]
    skip_out = set(a["name"] for a in F["args"] if a["kind"] == "out")        # scratch arrays are not needed by the frameworks
    for nm, arr in arrays.items():
        if nm in skip_out: continue
        if nm in ("src", "dst"): P[nm + "0"] = to_dev((arr - 1).astype(np.int64)); continue
        P[nm] = to_dev(arr)
    for nm, v in scalars.items():
        if nm == "eps": continue
        P[nm] = to_dev(np.array(v, dtype=np.float64))
    if not ALLOW_CPU:    # every array of the call must live on the GPU
        if FW == "torch": bad = [n for n, v in P.items() if not v.is_cuda]
        else: bad = [n for n, v in P.items() if list(v.devices())[0].platform not in ("gpu", "cuda")]
        if bad: raise RuntimeError("arrays not on the GPU: " + ",".join(bad))
    return P, c, arrays

# ---------------------------------------------------------------- signatures
def sig_of(x):
    a = to_np(x) if not isinstance(x, np.ndarray) else x
    return L.sig(np.atleast_1d(a))

def compare(cand, ref, fam=None):
    az = tuple(L.SPEC["families"][fam].get("analytic_zero", ())) if fam else ()
    return L.compare_flat(cand, ref, az)

# ---------------------------------------------------------------- callables
def make_adjoint(fam, P, c, variant, scope):
    gn = L.grad_names(fam, scope); gset = set(gn)
    if FW == "torch":
        leaves = {n: (P[n].clone().requires_grad_(True) if n in gset else P[n]) for n in P}
        def evaluate():
            for n in gn:
                g = leaves[n].grad
                if g is not None: g.zero_()
            loss = M.loss_fn(ops, fam, leaves, c)
            loss.backward()
            return loss
        if variant == "graph":
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): evaluate()
            torch.cuda.current_stream().wait_stream(s); sync()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): holder = evaluate()
            def run(): g.replay(); return holder
        elif variant == "compile":
            comp = torch.compile(lambda lv: M.loss_fn(ops, fam, lv, c))
            def evaluate_c():
                for n in gn:
                    gg = leaves[n].grad
                    if gg is not None: gg.zero_()
                l = comp(leaves); l.backward(); return l
            run = evaluate_c
        else: run = evaluate
        def result(): return run, {"loss": (lambda: None)}, leaves
        def grads(): return {n: leaves[n].grad for n in gn}
        def lossv(l): return l
        return run, grads, gn
    else:
        G = {n: P[n] for n in gn}; R = {n: v for n, v in P.items() if n not in gset}
        f = jax.jit(jax.value_and_grad(lambda G_, R_: M.loss_fn(ops, fam, {**R_, **G_}, c)))
        state = {}
        def run():
            out = f(G, R); state["out"] = out; return out
        def grads(): return state["out"][1]
        return run, grads, gn

def make_primal(fam, P, c, variant):
    if "x" in P and fam == "transformer": pass
    if FW == "torch":
        Pp = dict(P)
        def run():
            with torch.no_grad(): return M.FORWARD[fam](ops, Pp, c)
        if variant == "graph":
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): run()
            torch.cuda.current_stream().wait_stream(s); sync()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad(): holder = M.FORWARD[fam](ops, Pp, c)
            def runner(): g.replay(); return holder
            return runner
        return run
    f = jax.jit(lambda P_: M.FORWARD[fam](ops, P_, c))
    return lambda: f(P)

def block(x):
    if FW == "jax": jax.block_until_ready(x)
    else: sync()

# ---------------------------------------------------------------- timing
def time_protocol(run, warmup=10, trials=30, tmin=0.05, kmax=1000, nevent=30):
    warmup = TIME.get("warmup", warmup); trials = TIME.get("trials", trials); tmin = TIME.get("tmin", tmin); nevent = TIME.get("nevent", nevent)
    out = run(); block(out)                                    # first call (may compile)
    t0 = time.perf_counter(); out = run(); block(out); tq = time.perf_counter() - t0
    reduced = False
    if tq > 0.1: warmup = min(warmup, 2); reduced = True       # slow call: shorter warm-up
    if tq > 0.5: trials = int(min(trials, max(5, 15 / tq))); nevent = min(nevent, 5); reduced = True
    for _ in range(max(0, warmup - 2)): out = run()
    block(out)
    t1s = []
    for _ in range(3):
        t0 = time.perf_counter(); out = run(); block(out); t1s.append(time.perf_counter() - t0)
    K = int(min(max(math.ceil(tmin / max(sorted(t1s)[1], 1e-7)), 1), kmax))
    gc.collect(); gc.disable()
    try:
        tr = []
        for _ in range(trials):
            t0 = time.perf_counter_ns()
            for _ in range(K): out = run()
            block(out)
            tr.append((time.perf_counter_ns() - t0) / K / 1e3)
        ev = []
        if FW == "torch" and DEV == "cuda":
            for _ in range(nevent):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); out = run(); b.record(); torch.cuda.synchronize(); ev.append(a.elapsed_time(b) * 1e3)
        else:    # JAX has no verified event API here: wall time of one synchronized call (a different metric)
            for _ in range(nevent):
                block(out); t0 = time.perf_counter_ns(); out = run(); block(out); ev.append((time.perf_counter_ns() - t0) / 1e3)
    finally:
        gc.enable()
    return {"trial_us": tr, "event_us": ev, "K": K, "trials": trials, "warmup": warmup, "reduced_protocol": reduced}

# ---------------------------------------------------------------- memory
def gpu_free():
    if FW == "torch" and DEV == "cuda": return torch.cuda.mem_get_info()[0]
    return None

def jax_stats():
    try:
        ms = jax.devices()[0].memory_stats()
        return {k: ms.get(k) for k in ("bytes_in_use", "peak_bytes_in_use", "pool_bytes", "peak_pool_bytes")} if ms else {}
    except Exception: return {}

def nbytes(x):
    try: return int(x.nbytes) if FW == "jax" else int(x.element_size() * x.nelement())
    except Exception: return 0

def memory_probe(fam, params, mode, variant, scope):
    """Footprint of one gradient evaluation (or forward). Runs in the current process; see the plan section 8.2."""
    rec = {}
    gc.collect()
    if FW == "torch" and DEV == "cuda": torch.cuda.empty_cache(); sync(); rec["free0"] = gpu_free(); rec["alloc_base"] = torch.cuda.memory_allocated(); rec["reserved_base"] = torch.cuda.memory_reserved()
    if FW == "jax": rec["jax0"] = jax_stats()
    P, c, _ = build(fam, params, mode)
    if mode == "adjoint": run, grads, gn = make_adjoint(fam, P, c, variant, scope)
    else: run = make_primal(fam, P, c, variant); gn = []
    for _ in range(3): out = run()
    block(out)
    inputs = sum(nbytes(v) for v in P.values())
    rec["inputs_bytes"] = inputs
    if FW == "torch" and DEV == "cuda":
        rec["alloc_before"] = torch.cuda.memory_allocated(); rec["reserved_before_run"] = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats(); out = run(); sync()
        rec["peak_alloc"] = torch.cuda.max_memory_allocated(); rec["alloc_after"] = torch.cuda.memory_allocated(); rec["peak_reserved"] = torch.cuda.max_memory_reserved()
        rec["free1"] = gpu_free(); torch.cuda.empty_cache(); rec["free2"] = gpu_free()
        rec["M_A"] = rec["peak_alloc"] - rec["alloc_base"]       # net of allocations that existed before this probe (library handles, leftovers)
        rec["M_A_absolute"] = rec["peak_alloc"]
        rec["M_B"] = rec["peak_reserved"] - rec["reserved_base"]      # reserved by the caching allocator above the baseline
        rec["M_B_driver"] = rec["free0"] - rec["free1"]; rec["M_C"] = rec["free0"] - rec["free2"]
    elif FW == "jax":
        out = run(); block(out); rec["jax1"] = jax_stats()
        j0b = (rec.get("jax0") or {}).get("bytes_in_use") or 0; j0p = (rec.get("jax0") or {}).get("pool_bytes") or 0
        rec["M_A_absolute"] = rec["jax1"].get("peak_bytes_in_use")
        rec["M_A"] = (rec["M_A_absolute"] - j0b) if rec["M_A_absolute"] is not None else None      # net of the bytes in use before the probe (the peak is monotone: see section 8.2)
        rec["M_B"] = (rec["jax1"].get("pool_bytes") - j0p) if rec["jax1"].get("pool_bytes") is not None else None
        try:
            if mode == "adjoint":
                G = {n: P[n] for n in L.grad_names(fam, scope)}; R = {n: v for n, v in P.items() if n not in G}
                f = jax.jit(jax.value_and_grad(lambda G_, R_: M.loss_fn(ops, fam, {**R_, **G_}, c)))
                ma = f.lower(G, R).compile().memory_analysis()
            else:
                ma = jax.jit(lambda P_: M.FORWARD[fam](ops, P_, c)).lower(P).compile().memory_analysis()
            rec["static"] = {k: getattr(ma, k, None) for k in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "generated_code_size_in_bytes")}
        except Exception as e: rec["static_error"] = repr(e)[:200]
    if mode == "adjoint" and FW == "torch":
        rec["grads_bytes"] = sum(nbytes(P[n]) for n in gn)
    return rec

# ---------------------------------------------------------------- tasks
CONTENDERS = {"torch": {"eager": "P-EAGER", "graph": "P-GRAPH", "compile": "P-COMPILE"}, "jax": {"jit": "J-JIT"}}

def run_task(t):
    case = next(c for c in L.SPEC["cases"] if c["id"] == t["case"]); fam = case["family"]
    size = next(s for s in case["sizes"] if s["id"] == t["size"]); params = size["params"]
    modes = t.get("modes", ["gate"]); variants = t.get("variants") or (["eager", "graph"] if FW == "torch" else ["jit"])
    scopes = t.get("scopes") or ["all", "params"]
    for mode in [m for m in ("primal", "adjoint") if t.get(m, True)]:
        key = f"{t['case']}/{t['size']}/{mode}"
        for variant in variants:
            for scope in (scopes if mode == "adjoint" else ["all"]):
                name = CONTENDERS[FW][variant] + ("(params)" if scope == "params" else "")
                base = {"case": t["case"], "size": t["size"], "contender": name, "mode": mode, "scope": scope}
                try:
                    P, c, _ = build(fam, params, mode)
                    if mode == "adjoint": run, grads, gn = make_adjoint(fam, P, c, variant, scope)
                    else: run = make_primal(fam, P, c, variant); gn = []
                    ct0 = time.perf_counter(); out = run(); block(out); first_call_s = time.perf_counter() - ct0
                    if "gate" in modes and scope == "all":
                        for _ in range(2): out = run(); block(out)
                        if mode == "primal":
                            F = L.SPEC["families"][fam]; o = out if FW == "jax" else out
                            s = {F["gate_out_primal"][0]: sig_of(o)}
                            entry = {"out": s}
                        else:
                            entry = {"grads": {n: sig_of(g) for n, g in grads().items()}}
                            lv = out[0] if FW == "jax" else out
                            F = L.SPEC["families"][fam]
                            if F["loss"] is not None: entry["loss"] = sig_of(lv)
                        flat = entry["out"] if mode == "primal" else {**entry["grads"], **({"loss": entry["loss"]} if "loss" in entry else {})}
                        if variant in ("eager", "jit") and args.write_ref: NEWREF[key] = flat
                        ref = NEWREF.get(key) or REF.get(key)
                        errs = compare(flat, ref, fam) if ref else {}
                        mx = max(errs.values()) if errs else None
                        OUT["records"].append(dict(base, kind="gate", sigs=flat, errs_vs_ref=errs, max_err=mx, passed=(mx is not None and mx <= GATE_TOL) if ref else None))
                        if variant in ("eager", "jit"): NEWREF[key] = flat if key not in NEWREF else NEWREF[key]
                    if "time" in modes:
                        rec = time_protocol(run)
                        OUT["records"].append(dict(base, kind="time", first_call_s=first_call_s, **rec))
                    if "mem" in modes:
                        P = run = out = grads = gn = None
                        gc.collect()
                        if FW == "torch" and DEV == "cuda": torch.cuda.empty_cache()
                        OUT["records"].append(dict(base, kind="mem", **memory_probe(fam, params, mode, variant, scope)))
                except Exception as e:
                    OUT["errors"].append(dict(base, error=repr(e)[:500], trace=traceback.format_exc()[-900:]))
                    log(f"ERROR {base}: {e!r}")
                gc.collect()
                if FW == "torch" and DEV == "cuda": torch.cuda.empty_cache()


# ---------------------------------------------------------------- training (batch size 1)
def run_train(t):
    model = next(m for m in L.SPEC["train"]["models"] if m["id"] == t["model"]); fam = model["family"]
    size = next(s for s in model["sizes"] if s["id"] == t["size"]); params = size["params"]
    TR = L.SPEC["train"]; N, nwarm, nsteps, lr = TR["N"], TR["nwarm"], TR["nsteps"], TR["lr"].get(t["size"], TR["lr"][fam]); total = nwarm + nsteps
    F = L.SPEC["families"][fam]; ps = F["per_sample"]; pnames = [a["name"] for a in F["args"] if a.get("param")]
    P, c, _ = build(fam, params, "adjoint")
    ds = L.make_dataset(fam, params, N); lens = {nm: n for nm, (arr, n) in ds.items()}; dsd = {nm: to_dev(arr) for nm, (arr, n) in ds.items()}
    variant = t.get("variant", "eager" if FW == "torch" else "jit")
    base = {"model": t["model"], "size": t["size"], "contender": CONTENDERS[FW][variant], "mode": "train", "lr": lr}
    if FW == "torch":
        leaves = {n: P[n].clone().requires_grad_(True) for n in pnames}; Pt = dict(P); Pt.update(leaves)
        init = {n: leaves[n].detach().clone() for n in pnames}
        idx = torch.zeros(1, dtype=torch.long, device=DEV)
        ds2 = {nm: dsd[nm].reshape(N, lens[nm]) for nm in ps}
        plist = [leaves[n] for n in pnames]
        def body():
            for nm in ps: Pt[nm].copy_(ds2[nm].index_select(0, idx).reshape(Pt[nm].shape))
            for p in plist:
                if p.grad is not None: p.grad.zero_()
            loss = M.loss_fn(ops, fam, Pt, c); loss.backward()
            with torch.no_grad(): torch._foreach_add_(plist, [p.grad for p in plist], alpha=-lr)
            return loss.detach()
        def reset_params():
            with torch.no_grad():
                for n in pnames: leaves[n].copy_(init[n])
        if variant == "graph":
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): body()
            torch.cuda.current_stream().wait_stream(s); sync(); reset_params()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): holder = body()
            reset_params()
            def step(k): idx.fill_(k % N); g.replay(); return holder
        else:
            def step(k): idx.fill_(k % N); return body()
        def reset(): reset_params()
    else:
        params0 = {n: P[n] for n in pnames}; consts = {k: v for k, v in P.items() if k not in pnames and k not in ps}
        import functools
        @functools.partial(jax.jit, donate_argnums=0)
        def train_step(pr, dsx, k):
            x = {nm: jax.lax.dynamic_slice_in_dim(dsx[nm], (k % N) * lens[nm], lens[nm]) for nm in ps}
            loss, grads = jax.value_and_grad(lambda q: M.loss_fn(ops, fam, {**consts, **x, **q}, c))(pr)
            return jax.tree_util.tree_map(lambda a, b: a - lr * b, pr, grads), loss
        holder = {"p": {n: jnp.array(v) for n, v in params0.items()}}
        def step(k):
            holder["p"], loss = train_step(holder["p"], dsd, jnp.asarray(k)); return loss
        def reset(): holder["p"] = {n: jnp.array(v) for n, v in params0.items()}
    # synchronized pass: loss trajectory and per-step times
    reset(); losses, stimes = [], []; t_first = None
    for k in range(total):
        t0 = time.perf_counter_ns(); l = step(k); block(l); dt = (time.perf_counter_ns() - t0) / 1e3
        if k == 0: t_first = dt / 1e6
        losses.append(float(to_np(l))); 
        if k >= nwarm: stimes.append(dt)
    # pipelined pass: one synchronization at the end
    reset(); 
    for k in range(nwarm): l = step(k)
    block(l); t0 = time.perf_counter_ns(); ls = []
    for k in range(nwarm, total): ls.append(step(k))
    block(ls[-1]); tot = (time.perf_counter_ns() - t0) / 1e9
    rec = dict(base, kind="train", first_step_s=t_first, sync_step_us=stimes, pipelined_total_s=tot, throughput_steps_per_s=nsteps / tot,
               losses_at={str(k): losses[k - 1] for k in (1, 10, 100, 300, total) if k <= total}, finite=bool(np.all(np.isfinite(losses))), nsteps=nsteps, nwarm=nwarm)
    if FW == "torch" and DEV == "cuda": rec["peak_alloc"] = torch.cuda.max_memory_allocated()
    if FW == "jax": rec["jax_stats"] = jax_stats()
    OUT["records"].append(rec)

def main():
    OUT["env"] = {"python": sys.version.split()[0], "fw_version": (torch.__version__ if FW == "torch" else jax.__version__),
                  "device": (torch.cuda.get_device_name(0) if FW == "torch" and DEV == "cuda" else (str(jax.devices()[0]) if FW == "jax" else "cpu")),
                  "import_seconds": time.perf_counter() - T0}
    for t in TASKS["tasks"]:
        if DEADLINE and time.time() + float(t.get("needs_s", 0)) > DEADLINE:
            OUT["records"].append(dict(kind="skipped", task=t, reason="time budget")); continue
        try:
            run_train(t) if "model" in t else run_task(t)
        except Exception as e:
            OUT["errors"].append(dict(task=t, error=repr(e)[:500], trace=traceback.format_exc()[-900:])); log(f"ERROR task {t}: {e!r}")
        log(f"task done: {t.get('case') or t.get('model')} {t.get('size')} ({time.perf_counter() - T0:.0f} s)")
        try: json.dump(OUT, open(args.out, "w"))      # partial output survives a kill
        except Exception: pass
    if args.write_ref:
        ref = dict(REF); ref.update(NEWREF); json.dump(ref, open(args.ref, "w"))
    OUT["total_seconds"] = time.perf_counter() - T0
    json.dump(OUT, open(args.out, "w"))
    print(json.dumps({"records": len(OUT["records"]), "errors": len(OUT["errors"]), "seconds": OUT["total_seconds"]}))
main()
