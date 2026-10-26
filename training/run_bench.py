"""Runs every latency / peak-RSS measurement in a FRESH child process, sequentially (CPU otherwise idle)."""
import json, os, subprocess, sys, time
WD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WD)
import skejja_measure as fm

MD, OD = os.path.join(WD, "models"), os.path.join(WD, "onnx_ext")
CHILD = os.path.join(WD, "skejja_bench_child.py")
INP = os.path.join(WD, "bench_inputs.json")
ORIG = sys.argv[1]
N = 300
cands = [
    ("gliclass-edge-trim32k-ft", os.path.join(MD, "gliclass-edge-v32k-ft"), "edge-v32k-ft"),
    ("gliclass-edge-trim32k-L8-ft", os.path.join(MD, "gliclass-edge-v32k-L8-ft"), "edge-v32k-L8-ft"),
    ("gliclass-edge-trim32k-zs", os.path.join(MD, "gliclass-edge-v32k"), "edge-v32k-zs"),
    ("gliclass-edge-trim64k-zs", os.path.join(MD, "gliclass-edge-v64k"), "edge-v64k-zs"),
    ("gliclass-edge-orig", ORIG, "edge-orig"),
]
out = []
env = dict(os.environ, HF_HUB_DISABLE_XET="1")
for model, mdir, stem in cands:
    variants = [("torch-fp32", "torch", mdir)]
    for v, suf in (("onnx-fp32", "fp32"), ("onnx-int8", "int8"), ("onnx-int8dyn", "int8dyn")):
        pth = os.path.join(OD, f"{stem}.{suf}.onnx")
        if os.path.exists(pth):
            variants.append((v, "onnxruntime", pth))
    for variant, runtime, path in variants:
        wmb = (os.path.getsize(os.path.join(path, "model.safetensors")) if runtime == "torch" else
               os.path.getsize(path) + (os.path.getsize(path + ".data") if os.path.exists(path + ".data") else 0)) / 1e6
        rec = dict(model=model, variant=variant, runtime=runtime, weights_mb=wmb, latency={})
        rec["peak_rss_mb"] = fm.peak_rss_mb(CHILD, WD, "rss", runtime, path, mdir, 4, "choice", 20, INP, env=env)
        rec["peak_rss_mb_lean"] = fm.peak_rss_mb(CHILD, WD, "rss", runtime, path, mdir, 4, "choice", 20, INP,
                                                 env=dict(env, SKEJJA_LEAN="1"))
        for threads in (1, 4):
            for task in ("choice", "noul"):
                t0 = time.time()
                so = subprocess.run([sys.executable, CHILD, WD, "latency", runtime, path, mdir, str(threads), task, str(N), INP],
                                    capture_output=True, text=True, env=env)
                line = [l for l in so.stdout.splitlines() if l.startswith("RESULT=")]
                if not line:
                    rec["latency"][f"{task}_t{threads}"] = {"error": so.stderr[-500:]}
                else:
                    rec["latency"][f"{task}_t{threads}"] = json.loads(line[0][7:])
                print(model, variant, threads, task, rec["latency"][f"{task}_t{threads}"], round(time.time() - t0, 1), flush=True)
        out.append(rec)
        json.dump(out, open(os.path.join(WD, "bench_results_v2.json"), "w"), indent=1)
print("BENCH_DONE", len(out))
