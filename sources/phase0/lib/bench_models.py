"""Model definitions written once over an `ops` layer (JaxOps or TorchOps).
Each family exposes forward(ops, P, c) -> output array (flat) and loss(ops, P, c) -> scalar.
P maps names to framework arrays (flat, or 2-D for matvec). c holds integers and float constants."""
import math

class JaxOps:
    name = "jax"
    def __init__(self):
        import jax, jax.numpy as jnp
        from jax import lax
        self.jax, self.jnp, self.lax = jax, jnp, lax
    def tanh(self, a): return self.jnp.tanh(a)
    def relu(self, a): return self.jax.nn.relu(a)
    def sum(self, a): return self.jnp.sum(a)
    def matmul(self, a, b): return self.jnp.matmul(a, b)
    def reshape(self, a, shape): return self.jnp.reshape(a, shape)
    def swapaxes(self, a, i, j): return self.jnp.swapaxes(a, i, j)
    def concat(self, xs, axis): return self.jnp.concatenate(xs, axis=axis)
    def softmax(self, a): return self.jax.nn.softmax(a, axis=-1)
    def sqrt(self, a): return self.jnp.sqrt(a)
    def mean_last(self, a): return self.jnp.mean(a, axis=-1, keepdims=True)
    def index_add_rows(self, nrows, idx, vals): return self.jnp.zeros((nrows, vals.shape[1]), vals.dtype).at[idx].add(vals)
    def conv2d(self, x, w, b, pad):
        y = self.lax.conv_general_dilated(x, w, (1, 1), [(pad, pad), (pad, pad)], dimension_numbers=("NCHW", "OIHW", "NCHW"))
        return y + self.jnp.reshape(b, (1, -1, 1, 1))
    def maxpool2(self, x): return self.lax.reduce_window(x, -self.jnp.inf, self.lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID")
    def upsample2(self, x): return self.jnp.repeat(self.jnp.repeat(x, 2, axis=2), 2, axis=3)
    def scan_steps(self, step, u0, nstep):
        return self.lax.scan(lambda u, _: (step(u), None), u0, None, length=nstep)[0]

class TorchOps:
    name = "torch"
    def __init__(self):
        import torch, torch.nn.functional as F
        self.t, self.F = torch, F
    def tanh(self, a): return self.t.tanh(a)
    def relu(self, a): return self.t.relu(a)
    def sum(self, a): return self.t.sum(a)
    def matmul(self, a, b): return self.t.matmul(a, b)
    def reshape(self, a, shape): return self.t.reshape(a, shape)
    def swapaxes(self, a, i, j): return self.t.transpose(a, i, j)
    def concat(self, xs, axis): return self.t.cat(xs, dim=axis)
    def softmax(self, a): return self.t.softmax(a, dim=-1)
    def sqrt(self, a): return self.t.sqrt(a)
    def mean_last(self, a): return self.t.mean(a, dim=-1, keepdim=True)
    def index_add_rows(self, nrows, idx, vals): return self.t.zeros((nrows, vals.shape[1]), dtype=vals.dtype, device=vals.device).index_add(0, idx, vals)
    def conv2d(self, x, w, b, pad): return self.F.conv2d(x, w, b, padding=pad)
    def maxpool2(self, x): return self.F.max_pool2d(x, 2)
    def upsample2(self, x): return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    def scan_steps(self, step, u0, nstep):
        u = u0
        for _ in range(nstep): u = step(u)
        return u

# ------------------------------------------------------------------ forward functions
def f_dotprod(o, P, c): return o.sum(P["u"] * P["v"])

def f_stencil_loss(o, P, c):
    u = P["u"]; w = (u[:-2] - 2.0 * u[1:-1]) + u[2:]
    return o.sum(w * w)

def f_matvec_loss(o, P, c):
    v = o.matmul(P["a"], P["u"]); return o.sum(v * v)

def f_advection(o, P, c):
    cc, dx, dt = P["c"], P["dx"], P["dt"]
    def step(u):
        du = u[1:] - u[:-1]
        return o.concat([u[:1], u[1:] - (cc * dt * du) / dx], 0)
    return o.scan_steps(step, P["u"], c["i_nstep"])

def f_mlp1d(o, P, c):
    nh = c["n_h"]
    h1 = o.tanh(P["w1"] * P["x"][0] + P["b1"])
    h2 = o.tanh(o.matmul(o.reshape(P["w2"], (nh, nh)), h1) + P["b2"])
    out = o.sum(P["w3"] * h2) + P["b3"][0]
    return (out - P["y"][0]) ** 2

def _ln(o, r, g, b, eps):
    m = o.mean_last(r); var = o.mean_last((r - m) * (r - m))
    return (r - m) / o.sqrt(var + eps) * g + b

def f_transformer(o, P, c):
    n, h, dk, dff, L = c["n"], c["h"], c["dk"], c["dff"], c["n_layers"]; d = h * dk; eps = c["eps"]
    x = o.reshape(P["x_in"] if "x_in" in P else P["x"], (n, d)); inv = 1.0 / math.sqrt(dk)
    def sl(name, size, l): return P[name][l * size:(l + 1) * size]
    def heads(t): return o.swapaxes(o.reshape(t, (n, h, dk)), 0, 1)
    for l in range(L):
        q = o.matmul(x, o.reshape(sl("wq", d * d, l), (d, d))) + sl("bq", d, l)
        k = o.matmul(x, o.reshape(sl("wk", d * d, l), (d, d))) + sl("bk", d, l)
        v = o.matmul(x, o.reshape(sl("wv", d * d, l), (d, d))) + sl("bv", d, l)
        probs = o.softmax(o.matmul(heads(q), o.swapaxes(heads(k), 1, 2)) * inv)
        ctx = o.reshape(o.swapaxes(o.matmul(probs, heads(v)), 0, 1), (n, d))
        attn = o.matmul(ctx, o.reshape(sl("wo", d * d, l), (d, d))) + sl("bo", d, l)
        n1 = _ln(o, x + attn, sl("ln1_gain", d, l), sl("ln1_bias", d, l), eps)
        hid = o.relu(o.matmul(n1, o.reshape(sl("w1", d * dff, l), (d, dff))) + sl("b1", dff, l))
        ff = o.matmul(hid, o.reshape(sl("w2", dff * d, l), (dff, d))) + sl("b2", d, l)
        x = _ln(o, n1 + ff, sl("ln2_gain", d, l), sl("ln2_bias", d, l), eps)
    return o.reshape(x, (-1,))

def f_unet(o, P, c):
    h, w, ci, c1, c2, c3, co = (c[k] for k in ("h", "w", "c_in", "c1", "c2", "c3", "c_out"))
    def conv(x, wn, bn, nout, nin, k=3): return o.conv2d(x, o.reshape(P[wn], (nout, nin, k, k)), P[bn], k // 2)
    x = o.reshape(P["x"], (1, ci, h, w))
    t = o.relu(conv(x, "w_e1a", "b_e1a", c1, ci)); s1 = o.relu(conv(t, "w_e1b", "b_e1b", c1, c1)); p = o.maxpool2(s1)
    t = o.relu(conv(p, "w_e2a", "b_e2a", c2, c1)); s2 = o.relu(conv(t, "w_e2b", "b_e2b", c2, c2)); p = o.maxpool2(s2)
    t = o.relu(conv(p, "w_ba", "b_ba", c3, c2)); b = o.relu(conv(t, "w_bb", "b_bb", c3, c3))
    t = o.relu(conv(o.concat([o.upsample2(b), s2], 1), "w_d2a", "b_d2a", c2, c3 + c2)); d2 = o.relu(conv(t, "w_d2b", "b_d2b", c2, c2))
    t = o.relu(conv(o.concat([o.upsample2(d2), s1], 1), "w_d1a", "b_d1a", c1, c2 + c1)); d1 = o.relu(conv(t, "w_d1b", "b_d1b", c1, c1))
    return o.reshape(conv(d1, "w_out", "b_out", co, c1, 1), (-1,))

def f_mpnn(o, P, c):
    nn_, ne, nf, ef, mf = (c[k] for k in ("n_nodes", "n_edges", "n_node_feat", "n_edge_feat", "n_msg_feat"))
    x = o.reshape(P["node_feat"], (nn_, nf)); e = o.reshape(P["edge_feat"], (ne, ef))
    src, dst = P["src0"], P["dst0"]
    msg_in = o.concat([x[src], x[dst], e], 1)
    msg = o.relu(o.matmul(msg_in, o.swapaxes(o.reshape(P["w_msg"], (mf, 2 * nf + ef)), 0, 1)) + P["b_msg"])
    agg = o.index_add_rows(nn_, dst, msg)
    out = o.relu(o.matmul(o.concat([x, agg], 1), o.swapaxes(o.reshape(P["w_upd"], (nf, nf + mf)), 0, 1)) + P["b_upd"])
    return o.reshape(out, (-1,))

FORWARD = {"dotprod": f_dotprod, "stencil_loss": f_stencil_loss, "matvec_loss": f_matvec_loss, "advection": f_advection,
           "mlp1d": f_mlp1d, "transformer": f_transformer, "unet": f_unet, "mpnn": f_mpnn}
SCALAR_LOSS = {"dotprod", "stencil_loss", "matvec_loss", "mlp1d"}     # forward() already returns the scalar loss

def loss_fn(o, fam, P, c):
    """Scalar loss used for the gradient evaluation. Matches the loss of the STADE root kernel."""
    out = FORWARD[fam](o, P, c)
    if fam in SCALAR_LOSS: return out
    if fam == "advection": return o.sum(P["u_seed"] * out)
    nout = out.shape[0]; t = P["target"][:nout]
    return o.sum((out - t) * (out - t))
