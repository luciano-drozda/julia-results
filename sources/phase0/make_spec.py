"""Single source of truth for every benchmark case. Writes lib/cases_spec.json.
Both the Julia runner and the Python harnesses read this file, so data and argument order cannot drift.
Formulas use only + - * / ( ) sqrt and the integer parameters of the size."""
import json

def A(name, kind, length=None, **kw):
    d = {"name": name, "kind": kind}
    if length is not None: d["len"] = length
    d.update(kw); return d

_seed = [0]
def IN(name, length, scale="1", shift="0", shape=None, grad=False, param=False):
    _seed[0] += 1
    return A(name, "in", length, seed=_seed[0], scale=scale, shift=shift, shape=shape, grad=grad, param=param)
def OUT(name, length): return A(name, "out", length)
def INT(name, value=None): return A(name, "int", value=value or name)
def FS(name, value): return A(name, "fscalar", value=value, grad=True)
def IDX(name, length, rng): _seed[0] += 1; return A(name, "idx", length, seed=_seed[0], range=rng)

fam = {}

fam["dotprod"] = dict(kernel="dotprod", primal_kernel="dotprod", loss="loss", gate_out_primal=["loss"], gate_out_adjoint=["loss"],
    args=[OUT("loss", "1"), IN("u", "i_n", grad=True), IN("v", "i_n", grad=True), INT("i_n")], primal_args=None,
    zero_each=["loss"], reload=[], primal_zero_each=["loss"], primal_reload=[])

fam["stencil_loss"] = dict(kernel="stencil_loss", primal_kernel="stencil_loss", loss="loss", gate_out_primal=["loss"], gate_out_adjoint=["loss"],
    args=[OUT("loss", "1"), IN("u", "i_n", grad=True), OUT("w", "i_n"), INT("i_n")], primal_args=None,
    zero_each=["loss"], reload=[], primal_zero_each=["loss"], primal_reload=[])

fam["matvec_loss"] = dict(kernel="matvec_loss", primal_kernel="matvec_loss", loss="loss", gate_out_primal=["loss"], gate_out_adjoint=["loss"],
    args=[OUT("loss", "1"), IN("a", "i_m*i_n", shape=["i_m", "i_n"], grad=True), IN("u", "i_n", grad=True), OUT("v", "i_m"), INT("i_m"), INT("i_n")], primal_args=None,
    zero_each=["loss", "v"], reload=[], primal_zero_each=["loss", "v"], primal_reload=[])

fam["advection"] = dict(kernel="advection", primal_kernel="advection", loss=None, gate_out_primal=["u"], gate_out_adjoint=["u"],
    args=[IN("u", "i_nnode", grad=True), OUT("du", "i_nnode"), FS("c", "0.4"), FS("dx", "1.0"), FS("dt", "0.1"), INT("i_nstep"), INT("i_nnode")],
    primal_args=None, extra=[IN("u_seed", "i_nnode", scale="0.5")], seed_shadow={"u": "u_seed"},
    zero_each=[], reload=["u", "du"], primal_zero_each=[], primal_reload=["u", "du"])

fam["mlp1d"] = dict(kernel="mlp1d", primal_kernel="mlp1d", loss="loss", gate_out_primal=["loss"], gate_out_adjoint=["loss"],
    args=[OUT("loss", "1"), IN("x", "1", grad=True), IN("y", "1", grad=True), IN("w1", "n_h", grad=True, param=True),
          IN("b1", "n_h", scale="0.1", grad=True, param=True), IN("w2", "n_h*n_h", scale="1/sqrt(n_h)", grad=True, param=True),
          IN("b2", "n_h", scale="0.1", grad=True, param=True), IN("w3", "n_h", scale="1/sqrt(n_h)", grad=True, param=True),
          IN("b3", "1", scale="0.1", grad=True, param=True), OUT("h1", "n_h"), OUT("h2", "n_h"), OUT("o", "1"), INT("n_h")],
    primal_args=None, zero_each=["loss"], reload=[], primal_zero_each=["loss"], primal_reload=[])

# ---- transformer: base kernel (primal) and loss wrapper (adjoint)
W = lambda nm, ln, sc: IN(nm, ln, scale=sc, grad=True, param=True)
tr_args = [OUT("x", "n*d"),
    W("wq", "n_layers*d*d", "1/sqrt(d)"), W("bq", "n_layers*d", "0.1"), W("wk", "n_layers*d*d", "1/sqrt(d)"), W("bk", "n_layers*d", "0.1"),
    W("wv", "n_layers*d*d", "1/sqrt(d)"), W("bv", "n_layers*d", "0.1"), W("wo", "n_layers*d*d", "1/sqrt(d)"), W("bo", "n_layers*d", "0.1"),
    IN("ln1_gain", "n_layers*d", scale="0.1", shift="1", grad=True, param=True), W("ln1_bias", "n_layers*d", "0.1"),
    W("w1", "n_layers*d*dff", "1/sqrt(d)"), W("b1", "n_layers*dff", "0.1"), W("w2", "n_layers*dff*d", "1/sqrt(dff)"), W("b2", "n_layers*d", "0.1"),
    IN("ln2_gain", "n_layers*d", scale="0.1", shift="1", grad=True, param=True), W("ln2_bias", "n_layers*d", "0.1")]
for nm, ln in (("q", "n*d"), ("k", "n*d"), ("v", "n*d"), ("scores", "h*n*n"), ("probs", "h*n*n"), ("ctx", "n*d"), ("attn_out", "n*d"),
               ("resid1", "n*d"), ("normed1", "n*d"), ("ff_hidden", "n*dff"), ("ff_out", "n*d"), ("resid2", "n*d"), ("x_next", "n*d")):
    tr_args.append(OUT(nm, ln))
tr_args += [INT("n"), INT("dk"), INT("h"), INT("dff"), INT("n_layers"), FS("eps", "0.00001")]
tr_args[-1]["grad"] = False
base_names = [a["name"] for a in tr_args]
tr_adj = [dict(a) for a in tr_args]
tr_adj += [IN("x_in", "n*d", grad=True), IN("target", "n*d", grad=True), OUT("loss", "1")]
fam["transformer"] = dict(kernel="transformer_loss", primal_kernel="transformer", loss="loss", gate_out_primal=["x"], gate_out_adjoint=["loss"],
    args=tr_adj, primal_args=base_names, primal_alias={"x": "x_in"},
    zero_each=["loss"], reload=[], primal_zero_each=[], primal_reload=["x"])
fam["transformer"]["primal_arg_specs"] = tr_args
fam["transformer"]["analytic_zero"] = ["bk"]      # softmax cancels a constant shift of the keys: d loss / d bk == 0 exactly

# ---- unet
def conv_w(nm, co, ci, k="9"): return IN(nm, f"{co}*{ci}*{k}", scale=f"1/sqrt({ci}*{k})", grad=True, param=True)
def bias(nm, co): return IN(nm, co, scale="0.1", grad=True, param=True)
un_args = [IN("x", "c_in*h*w", grad=True), INT("h"), INT("w"), INT("c_in"), INT("c1"), INT("c2"), INT("c3"), INT("c_out"),
    conv_w("w_e1a", "c1", "c_in"), bias("b_e1a", "c1"), conv_w("w_e1b", "c1", "c1"), bias("b_e1b", "c1"),
    conv_w("w_e2a", "c2", "c1"), bias("b_e2a", "c2"), conv_w("w_e2b", "c2", "c2"), bias("b_e2b", "c2"),
    conv_w("w_ba", "c3", "c2"), bias("b_ba", "c3"), conv_w("w_bb", "c3", "c3"), bias("b_bb", "c3"),
    conv_w("w_d2a", "c2", "(c3+c2)"), bias("b_d2a", "c2"), conv_w("w_d2b", "c2", "c2"), bias("b_d2b", "c2"),
    conv_w("w_d1a", "c1", "(c2+c1)"), bias("b_d1a", "c1"), conv_w("w_d1b", "c1", "c1"), bias("b_d1b", "c1"),
    conv_w("w_out", "c_out", "c1", "1"), bias("b_out", "c_out")]
hw, hw2, hw4 = "h*w", "(h/2)*(w/2)", "(h/4)*(w/4)"
pad1, pad2, pad4 = "(h+2)*(w+2)", "((h/2)+2)*((w/2)+2)", "((h/4)+2)*((w/4)+2)"
for nm, ln in (("xpad0", f"c_in*{pad1}"), ("t_e1", f"c1*{hw}"), ("t_e1pad", f"c1*{pad1}"), ("skip1", f"c1*{hw}"), ("p1", f"c1*{hw2}"), ("p1pad", f"c1*{pad2}"),
               ("t_e2", f"c2*{hw2}"), ("t_e2pad", f"c2*{pad2}"), ("skip2", f"c2*{hw2}"), ("p2", f"c2*{hw4}"), ("p2pad", f"c2*{pad4}"),
               ("t_b", f"c3*{hw4}"), ("t_bpad", f"c3*{pad4}"), ("bott", f"c3*{hw4}"),
               ("u2", f"c3*{hw2}"), ("cat2", f"(c3+c2)*{hw2}"), ("cat2pad", f"(c3+c2)*{pad2}"), ("t_d2", f"c2*{hw2}"), ("t_d2pad", f"c2*{pad2}"), ("dec2out", f"c2*{hw2}"),
               ("u1", f"c2*{hw}"), ("cat1", f"(c2+c1)*{hw}"), ("cat1pad", f"(c2+c1)*{pad1}"), ("t_d1", f"c1*{hw}"), ("t_d1pad", f"c1*{pad1}"), ("dec1out", f"c1*{hw}"),
               ("y", f"c_out*{hw}")):
    un_args.append(OUT(nm, ln))
un_base = [a["name"] for a in un_args]
un_adj = [dict(a) for a in un_args] + [IN("target", f"c_out*{hw}", grad=True), OUT("loss", "1")]
fam["unet"] = dict(kernel="unet_loss", primal_kernel="unet", loss="loss", gate_out_primal=["y"], gate_out_adjoint=["loss"],
    args=un_adj, primal_args=un_base, zero_each=["loss"], reload=[], primal_zero_each=[], primal_reload=[])
fam["unet"]["primal_arg_specs"] = un_args

# ---- mpnn
mp_args = [IN("node_feat", "n_nodes*n_node_feat", grad=True), IN("edge_feat", "n_edges*n_edge_feat", grad=True),
    IDX("src", "n_edges", "n_nodes"), IDX("dst", "n_edges", "n_nodes"),
    IN("w_msg", "n_msg_feat*(2*n_node_feat+n_edge_feat)", scale="1/sqrt(2*n_node_feat+n_edge_feat)", grad=True, param=True), IN("b_msg", "n_msg_feat", scale="0.1", grad=True, param=True),
    IN("w_upd", "n_node_feat*(n_node_feat+n_msg_feat)", scale="1/sqrt(n_node_feat+n_msg_feat)", grad=True, param=True), IN("b_upd", "n_node_feat", scale="0.1", grad=True, param=True),
    INT("n_nodes"), INT("n_edges"), INT("n_node_feat"), INT("n_edge_feat"), INT("n_msg_feat"),
    OUT("msg_input", "n_edges*(2*n_node_feat+n_edge_feat)"), OUT("msg_scratch", "n_edges*n_msg_feat"), OUT("messages", "n_edges*n_msg_feat"),
    OUT("agg", "n_nodes*n_msg_feat"), OUT("upd_input", "n_nodes*(n_node_feat+n_msg_feat)"), OUT("upd_scratch", "n_nodes*n_node_feat"),
    OUT("node_feat_out", "n_nodes*n_node_feat")]
mp_base = [a["name"] for a in mp_args]
mp_adj = [dict(a) for a in mp_args] + [IN("target", "n_nodes*n_node_feat", grad=True), OUT("loss", "1")]
fam["mpnn"] = dict(kernel="mpnn_loss", primal_kernel="mpnn", loss="loss", gate_out_primal=["node_feat_out"], gate_out_adjoint=["loss"],
    args=mp_adj, primal_args=mp_base, zero_each=["loss"], reload=[], primal_zero_each=[], primal_reload=[])
fam["mpnn"]["primal_arg_specs"] = mp_args

# ---- sizes
def P(**kw): return kw
cases = []
def case(cid, family, label, sizes, gate=None): cases.append(dict(id=cid, family=family, label=label, sizes=sizes, gate=gate))
case("K1", "dotprod", "dot product", [dict(id=f"n2^{e}", params=P(i_n=2**e)) for e in (10, 14, 15, 16, 18, 20, 24, 26)], gate=["n2^10", "n2^16", "n2^26"])
case("K2", "stencil_loss", "stencil loss", [dict(id=f"n2^{e}", params=P(i_n=2**e)) for e in (12, 16, 20, 24, 26)], gate=["n2^12", "n2^20", "n2^26"])
case("K3", "matvec_loss", "matrix-vector loss", [dict(id=f"m{m}", params=P(i_m=m, i_n=m)) for m in (256, 1024, 4096, 8192, 12288)], gate=["m256", "m4096", "m12288"])
case("K4", "advection", "advection", [dict(id=f"{nn}x{ns}", params=P(i_nnode=nn, i_nstep=ns)) for nn, ns in ((1000, 100), (1000, 1000), (100000, 100), (100000, 1000), (1000000, 100))], gate=["1000x100", "100000x100", "1000000x100"])
case("K5", "mlp1d", "MLP (scalar regression)", [dict(id=f"nh{n}", params=P(n_h=n)) for n in (32, 128, 512, 2048, 4096)], gate=["nh32", "nh512", "nh4096"])
TS, TM, TL = dict(n=16, h=4, dk=16, dff=128, n_layers=2), dict(n=64, h=8, dk=32, dff=512, n_layers=4), dict(n=128, h=8, dk=64, dff=1024, n_layers=6)
tsizes = [dict(id="T-S", params=TS), dict(id="T-M", params=TM), dict(id="T-L", params=TL)] + [dict(id=f"T-M-n{n}", params=dict(TM, n=n)) for n in (16, 32, 128, 256)]
case("K6", "transformer", "transformer encoder", tsizes, gate=["T-S", "T-M", "T-L"])
usz = [dict(id=f"narrow-{s}", params=P(h=s, w=s, c_in=3, c1=8, c2=16, c3=32, c_out=2)) for s in (16, 32, 64, 128, 256)] + \
      [dict(id=f"wide-{s}", params=P(h=s, w=s, c_in=3, c1=16, c2=32, c3=64, c_out=2)) for s in (32, 64)]
case("K7", "unet", "U-Net", usz, gate=["narrow-16", "narrow-64", "narrow-256"])
case("K8", "mpnn", "message-passing network", [dict(id=f"nn{n}", params=P(n_nodes=n, n_edges=8*n, n_node_feat=16, n_edge_feat=8, n_msg_feat=16)) for n in (1000, 10000, 100000)], gate=["nn1000", "nn10000", "nn100000"])
# tiny sizes for the preflight job J0 and for tests
tiny = {"K1": P(i_n=1000), "K2": P(i_n=1000), "K3": P(i_m=40, i_n=30), "K4": P(i_nnode=200, i_nstep=7), "K5": P(n_h=16),
        "K6": dict(n=6, h=2, dk=4, dff=12, n_layers=2), "K7": P(h=8, w=8, c_in=3, c1=4, c2=6, c3=8, c_out=2), "K8": P(n_nodes=30, n_edges=100, n_node_feat=4, n_edge_feat=3, n_msg_feat=5)}
for c in cases: c["sizes"].insert(0, dict(id="tiny", params=tiny[c["id"]]))
derived = {"transformer": {"d": "h*dk"}}
for f, dv in derived.items(): fam[f]["derived"] = dv
# ---- training (batch size 1, plan section 9)
per_sample = {"mlp1d": ["x", "y"], "transformer": ["x_in", "target"], "unet": ["x", "target"], "mpnn": ["node_feat", "edge_feat", "target"]}
for f, ps in per_sample.items(): fam[f]["per_sample"] = ps
train_lr = {"mlp1d": 0.002, "transformer": 0.0005, "T-M": 0.00003, "unet": 0.0003, "mpnn": 0.001}   # a size id overrides the family default
TRAIN = dict(nwarm=20, nsteps=300, N=1024, lr=train_lr,
             models=[dict(id="M1", family="mlp1d", sizes=[dict(id=f"nh{n}", params=dict(n_h=n)) for n in (64, 512, 2048)]),
                     dict(id="M2", family="transformer", sizes=[dict(id="T-S", params=TS), dict(id="T-M", params=TM)]),
                     dict(id="M3", family="unet", sizes=[dict(id="narrow-32", params=dict(h=32, w=32, c_in=3, c1=8, c2=16, c3=32, c_out=2))]),
                     dict(id="M4", family="mpnn", sizes=[dict(id="nn1000", params=dict(n_nodes=1000, n_edges=8000, n_node_feat=16, n_edge_feat=8, n_msg_feat=16))])])
TRAIN["models"][0]["sizes"].insert(0, dict(id="tiny", params=dict(n_h=16)))
TRAIN["models"][1]["sizes"].insert(0, dict(id="tiny", params=tiny["K6"]))
TRAIN["models"][2]["sizes"].insert(0, dict(id="tiny", params=tiny["K7"]))
TRAIN["models"][3]["sizes"].insert(0, dict(id="tiny", params=tiny["K8"]))
EXCL = {"cases": {"K5": "mlp1d: the generated code runs the output-layer sum on the host below 32768 elements", "K6": "transformer: the generated code runs the softmax and LayerNorm row loops on the host"},
        "train_models": {"M1": "uses the mlp1d kernel (K5)", "M2": "uses the transformer kernel (K6)"}, "decision": "D15 (c)"}
spec = {"seed_base": 12345, "families": fam, "cases": cases, "train": TRAIN, "excluded": EXCL}
json.dump(spec, open("lib/cases_spec.json", "w"), indent=1)
print("families:", list(fam), "| cases:", [(c["id"], len(c["sizes"])) for c in cases])
