"""Generate methods_baseline.md from metrics_baseline.csv (all numbers are read from the CSV)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
WS = os.path.dirname(HERE)
import pandas as pd
import baseline_lib as bl

M = {"laya": "laya-multilingual", "gli": "gliclass-multilang-edge"}
VARS = ["torch-fp32", "onnx-fp32", "onnx-int8", "onnx-w8e4", "onnx-w8e4a4"]


def main(csv="metrics_baseline.csv", out="methods_baseline.md", extra=None):
    d = pd.read_csv(os.path.join(WS, csv))
    extra = extra or {}

    def v(model, variant, task, metric, split="all", lang="all", threads=None):
        s = d[(d.model == model) & (d.variant == variant) & (d.task == task) & (d.metric == metric) & (d.split == split) & (d.lang == lang)]
        if threads is not None:
            s = s[s.threads == threads]
        return float(s.value.iloc[0]) if len(s) else float("nan")

    f = lambda x, n=3: ("%." + str(n) + "f") % x if x == x else "n/a"
    # ---------------- summary table
    hdr = ("| model | variant | weights MB | choice acc (macro) | worst-lang gap vs en | choice ECE raw/cal (pooled) | "
           "noul AUROC (macro) | noul ECE raw/cal (pooled) | choice p50/p95 ms @4t | choice p50 ms @1t | noul p50 ms @4t | "
           "peak RSS MB (choice) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [hdr]
    for k, mname in M.items():
        for var in VARS:
            if not ((d.model == mname) & (d.variant == var)).any():
                continue
            accs = {l: v(mname, var, "choice", "accuracy", lang=l) for l in ["en", "it", "es", "de", "fr"]}
            gap = accs["en"] - min(accs[l] for l in ["it", "es", "de", "fr"])
            lines.append("| %s | %s | %s | %s | %s | %s / %s | %s | %s / %s | %s / %s | %s | %s | %s |" % (
                mname, var, f(v(mname, var, "choice", "weights_mb", "memory"), 1),
                f(v(mname, var, "choice", "accuracy", lang="macro")), f(gap),
                f(v(mname, var, "choice", "ece_raw")), f(v(mname, var, "choice", "ece_cal")),
                f(v(mname, var, "noul", "auroc", lang="macro")),
                f(v(mname, var, "noul", "ece_raw")), f(v(mname, var, "noul", "ece_cal")),
                f(v(mname, var, "choice", "p50_ms", "latency", threads=4), 1), f(v(mname, var, "choice", "p95_ms", "latency", threads=4), 1),
                f(v(mname, var, "choice", "p50_ms", "latency", threads=1), 1),
                f(v(mname, var, "noul", "p50_ms", "latency", threads=4), 1),
                f(v(mname, var, "choice", "peak_rss_mb", "memory"), 0)))
    table = "\n".join(lines)
    # ---------------- int8 deltas
    dl = []
    for k, mname in M.items():
        for var in ["onnx-int8", "onnx-w8e4", "onnx-w8e4a4"]:
            dl.append("- %s %s vs torch-fp32: choice accuracy %+.1f pts (macro), noul AUROC %+.3f (macro), noul accuracy@0.5 %+.1f pts; "
                      "decision agreement with torch on the full test set: choice %s, noul %s." % (
                          mname, var,
                          100 * (v(mname, var, "choice", "accuracy", lang="macro") - v(mname, "torch-fp32", "choice", "accuracy", lang="macro")),
                          v(mname, var, "noul", "auroc", lang="macro") - v(mname, "torch-fp32", "noul", "auroc", lang="macro"),
                          100 * (v(mname, var, "noul", "accuracy", lang="macro") - v(mname, "torch-fp32", "noul", "accuracy", lang="macro")),
                          f(v(mname, var, "choice", "agree_argmax_vs_torch_fulltest")), f(v(mname, var, "noul", "agree_argmax_vs_torch_fulltest"))))
    deltas = "\n".join(dl)
    agree_fp32 = "; ".join("%s %s: max|dlogit| %.1e, max|dp| %.1e, argmax agreement %s" % (
        mname, t, v(mname, "onnx-fp32", t, "agree_max_abs_diff_logit"), v(mname, "onnx-fp32", t, "agree_max_abs_diff_prob"),
        f(v(mname, "onnx-fp32", t, "agree_argmax_vs_torch"))) for mname in M.values() for t in ["choice", "noul"])
    txt = open(os.path.join(HERE, "methods_template.md")).read()
    rep = {"TABLE": table, "DELTAS": deltas, "AGREE_FP32": agree_fp32,
           "LAYA_T": f(v(M["laya"], "torch-fp32", "choice", "temperature")), "GLI_T": f(v(M["gli"], "torch-fp32", "choice", "temperature")),
           "LAYA_PA": f(v(M["laya"], "torch-fp32", "noul", "platt_a")), "LAYA_PB": f(v(M["laya"], "torch-fp32", "noul", "platt_b")),
           "GLI_PA": f(v(M["gli"], "torch-fp32", "noul", "platt_a")), "GLI_PB": f(v(M["gli"], "torch-fp32", "noul", "platt_b"))}
    rep.update(extra)
    for kk, vv in rep.items():
        txt = txt.replace("{" + kk + "}", str(vv))
    open(os.path.join(WS, out), "w").write(txt)
    return txt


if __name__ == "__main__":
    print(main()[:500])
