"""Assemble metrics_baseline.csv from out/quality_rows.csv + out/perf_results.jsonl + file sizes."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "bench"))   # skejja_measure
WS = os.path.dirname(HERE)
import pandas as pd
import skejja_measure as fm
import baseline_lib as bl

MODELS = {"laya": "laya-multilingual", "gli": "gliclass-multilang-edge"}
WEIGHTS = {
    ("laya", "torch-fp32"): os.path.join(bl.LAYA_DIR, "model.safetensors"),
    ("gli", "torch-fp32"): os.path.join(bl.GLI_DIR, "model.safetensors"),
    ("laya", "onnx-fp32"): "out/onnx/laya/laya_multilingual_fp32.onnx",
    ("laya", "onnx-int8"): "out/onnx/laya/laya_multilingual_int8.onnx",
    ("laya", "onnx-w8e4"): "out/onnx/laya/laya_multilingual_w8e4.onnx",
    ("gli", "onnx-fp32"): "out/onnx/gliclass/gliclass_multilang_edge_fp32.onnx",
    ("gli", "onnx-int8"): "out/onnx/gliclass/gliclass_multilang_edge_int8.onnx",
    ("gli", "onnx-w8e4"): "out/onnx/gliclass/gliclass_multilang_edge_w8e4.onnx",
}
PARAMS_M = {"laya": 321.9, "gli": 143.0}   # counted from the safetensors headers (see methods)


def build(perf_path="out/perf_results.jsonl", out_csv="metrics_baseline.csv", extra_variants=None):
    q = pd.read_csv(os.path.join(WS, "out/quality_rows.csv"))
    rows = q.to_dict("records")
    perf = [json.loads(l) for l in open(os.path.join(WS, perf_path)) if l.strip()]
    weights = dict(WEIGHTS)
    if extra_variants:
        weights.update(extra_variants)
    for r in perf:
        if "error" in r:
            continue
        rt = "torch" if r["variant"].startswith("torch") else "onnxruntime"
        if r["kind"] == "latency":
            rows += fm.rows(fm_track(), MODELS[r["model"]], r["variant"], rt, r["threads"], r["task"], "latency", "all",
                            {"p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"], "mean_ms": r["mean_ms"]}, r["n"])
        else:
            rows += fm.rows(fm_track(), MODELS[r["model"]], r["variant"], rt, r["threads"], r["task"], "memory", "all",
                            {"peak_rss_mb": r["peak_rss_mb"]}, 20)
    for (m, v), p in weights.items():
        p = p if os.path.isabs(p) else os.path.join(WS, p)
        rt = "torch" if v.startswith("torch") else "onnxruntime"
        for task in ("choice", "noul"):
            met = {"weights_mb": os.path.getsize(p) / 1e6}
            if v.startswith("torch"):
                met["params_m"] = PARAMS_M[m]
            rows += fm.rows(fm_track(), MODELS[m], v, rt, 4, task, "memory", "all", met, 1)
    df = pd.DataFrame(rows, columns=fm.METRIC_COLUMNS)
    df.to_csv(os.path.join(WS, out_csv), index=False)
    return df


def fm_track():
    return "baseline"


if __name__ == "__main__":
    d = build()
    print(d.shape)
