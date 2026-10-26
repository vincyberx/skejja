"""Sequentially run every latency / peak-RSS child (nothing else should run on the machine meanwhile).
Appends one JSON line per measurement to out/perf_results.jsonl."""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
WS = os.path.dirname(HERE)
import skejja_measure as fm

ONNX = {("laya", "onnx-fp32"): "out/onnx/laya/laya_multilingual_fp32.onnx",
        ("laya", "onnx-int8"): "out/onnx/laya/laya_multilingual_int8.onnx",
        ("laya", "onnx-w8e4"): "out/onnx/laya/laya_multilingual_w8e4.onnx",
        ("gli", "onnx-fp32"): "out/onnx/gliclass/gliclass_multilang_edge_fp32.onnx",
        ("gli", "onnx-int8"): "out/onnx/gliclass/gliclass_multilang_edge_int8.onnx",
        ("gli", "onnx-w8e4"): "out/onnx/gliclass/gliclass_multilang_edge_w8e4.onnx",
        ("laya", "onnx-w8e4a4"): "out/onnx/laya/laya_multilingual_w8e4a4.onnx",
        ("gli", "onnx-w8e4a4"): "out/onnx/gliclass/gliclass_multilang_edge_w8e4a4.onnx"}
CHILD = os.path.join(HERE, "lat_child.py")
OUT = os.path.join(WS, "out", "perf_results.jsonl")
only = sys.argv[1] if len(sys.argv) > 1 else "all"
VARIANTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["torch-fp32", "onnx-fp32", "onnx-int8", "onnx-w8e4"]


def cfgs():
    for model in ["gli", "laya"]:
        for variant in VARIANTS:
            for task in ["choice", "noul"]:
                yield model, variant, task


def args_for(model, variant, task, threads, mode):
    rt = "torch" if variant.startswith("torch") else "ort"
    a = [model, rt, task, threads, mode]
    if rt == "ort":
        a.append(os.path.join(WS, ONNX[(model, variant)]))
    return a


with open(OUT, "a") as fo:
    if only in ("all", "latency"):
        for model, variant, task in cfgs():
            for threads in (4, 1):
                t0 = time.time()
                p = subprocess.run([sys.executable, CHILD, *map(str, args_for(model, variant, task, threads, "latency"))],
                                   capture_output=True, text=True)
                line = [l for l in p.stdout.splitlines() if l.startswith("LATENCY_JSON=")]
                rec = dict(kind="latency", model=model, variant=variant, task=task, threads=threads, wall_s=round(time.time() - t0, 1))
                if line:
                    rec.update(json.loads(line[-1].split("=", 1)[1]))
                else:
                    rec["error"] = p.stderr[-2000:]
                fo.write(json.dumps(rec) + "\n"); fo.flush()
                print(json.dumps(rec), flush=True)
    if only in ("all", "rss"):
        for model, variant, task in cfgs():
            rec = dict(kind="rss", model=model, variant=variant, task=task, threads=4)
            try:
                rec["peak_rss_mb"] = fm.peak_rss_mb(CHILD, *args_for(model, variant, task, 4, "rss"))
            except Exception as e:
                rec["error"] = repr(e)[-2000:]
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(json.dumps(rec), flush=True)
