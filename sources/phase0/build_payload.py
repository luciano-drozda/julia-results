"""Assemble a job payload (single Julia script) from the library, the generated sources, and a job spec."""
import json, base64, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); LIB = os.path.join(HERE, "lib"); FINAL = os.path.join(HERE, "final")
b64 = lambda s: base64.b64encode(s if isinstance(s, bytes) else s.encode()).decode()
rd = lambda p: open(p).read()

def sources_for(job):
    """Names of generated files a job needs -> contents (stripped)."""
    spec = json.load(open(os.path.join(LIB, "cases_spec.json"))); need = {}
    target = job.get("target", "gpu")
    fams = set()
    for g in job["groups"]:
        for t in g["tasks"]:
            if "model" in t: fams.add(next(m for m in spec["train"]["models"] if m["id"] == t["model"])["family"])
            else: fams.add(next(c for c in spec["cases"] if c["id"] == t["case"])["family"])
    if job.get("probes"): fams.update(["mlp1d"])
    for f in sorted(fams):
        F = spec["families"][f]
        if target == "gpu":
            for be in ("cuda", "jacc"):
                for nm in {F["kernel"] + "_b_" + be + ".jl", (F["kernel"] + "_b_" + be + ".jl") if F["primal_kernel"] == F["kernel"] else (F["primal_kernel"] + "_" + be + ".jl")}:
                    need[nm] = rd(os.path.join(FINAL, nm.replace(".jl", ".stripped.jl")))
        else:
            need[F["kernel"] + "_b.jl"] = rd(os.path.join("/home/claude/work/gen_044", F["kernel"] + "_b.jl"))
            need[F["primal_kernel"] + ".jl"] = rd(os.path.join("/home/claude/work/stade_044/STADE.jl/test/val-corpus", F["primal_kernel"] + ".jl"))
        if job.get("kind") == "train" or job.get("probes"):
            if f in ("mlp1d", "transformer", "unet", "mpnn"):
                nm = F["kernel"] + "_batch.jl"; need[nm] = rd(os.path.join(FINAL, "batch", nm))
    return need

def build(job):
    target = job.get("target", "gpu")
    spec_text = rd(os.path.join(LIB, "cases_spec.json"))
    py = {n: rd(os.path.join(LIB, n)) for n in ("bench_lib.py", "bench_models.py", "harness_fw.py", "probe_fw.py")}
    py["cases_spec.json"] = spec_text
    src = sources_for(job)
    head = ["using JSON3, Base64, Dates, Pkg"]
    if target == "gpu": head.insert(0, "using CUDA, JACC")
    if job.get("use_mpi") and not job.get("stub_mpi"): head.append("using MPI")
    parts = ["\n".join(head), ""]
    parts.append(f'const SPEC_B64 = "{b64(spec_text)}"')
    parts.append(f'const JOB_B64 = "{b64(json.dumps(job))}"')
    parts.append("const PY_B64 = Dict{String,String}(" + ", ".join(f'"{n}" => "{b64(t)}"' for n, t in py.items()) + ")")
    parts.append("const SRC_B64 = Dict{String,String}(" + ", ".join(f'"{n}" => "{b64(t)}"' for n, t in sorted(src.items())) + ")")
    runner = rd(os.path.join(LIB, "stade_runner.jl")).replace("using JSON3\n", "")
    parts += [runner, rd(os.path.join(LIB, "backends_gpu.jl" if target == "gpu" else "backends_cpu.jl")),
              rd(os.path.join(LIB, "driver.jl")), rd(os.path.join(LIB, "train.jl")), rd(os.path.join(LIB, "probes.jl")), "main()"]
    return "\n".join(parts), sorted(src)

if __name__ == "__main__":
    job = json.load(open(sys.argv[1])); script, names = build(job)
    out = sys.argv[2]; open(out + ".jl", "w").write(script)
    json.dump({"input": {"_script": script, "_topology": "gpu_v100"}}, open(out + ".json", "w"))
    print("script bytes:", len(script), "| generated files embedded:", len(names))
