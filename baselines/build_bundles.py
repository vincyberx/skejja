"""Build per-(model, variant) tar.gz bundles: ONNX + tokenizer + torch-free runner + model_info.json + README."""
import json, os, sys, tarfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
WS = os.path.dirname(HERE)
import pandas as pd
import baseline_lib as bl

SPEC = {
    "laya": dict(name="laya-multilingual", repo=bl.LAYA_REPO, rev=bl.LAYA_REV, authors="Convai Innovations",
                 tok_files=[("tokenizer/tokenizer.json", "tokenizer.json"), ("tokenizer/tokenizer_config.json", "tokenizer_config.json")],
                 base=bl.LAYA_DIR, cls="LayaORT"),
    "gli": dict(name="gliclass-multilang-edge", repo=bl.GLI_REPO, rev=bl.GLI_REV, authors="Knowledgator",
                tok_files=[("tokenizer.json", "tokenizer.json"), ("tokenizer_config.json", "tokenizer_config.json"),
                           ("special_tokens_map.json", "special_tokens_map.json")],
                base=bl.GLI_DIR, cls="GLiClassORT"),
}
QUANT = {
    "onnx-fp32": "",
    "onnx-int8": " + onnxruntime.quantization.quantize_dynamic(weight_type=QInt8, per_channel=False; MatMul->MatMulInteger "
                 "with dynamic per-tensor uint8 activations, embedding Gather -> uint8)",
    "onnx-w8e4a4": " + weight-only quantization: MatMul -> MatMulNBits(bits=8, block_size=128, symmetric, accuracy_level=4), "
                   "embedding Gather -> GatherBlockQuantized(bits=4, block_size=32, symmetric)",
}
IO = {
    "laya": {"inputs": {"input_ids": "int64 [B,S]", "attention_mask": "int64 [B,S]", "marker_pos": "int64 [B,K]",
                        "marker_mask": "bool [B,K]", "qtype": "int64 [B] (choice=0, score=1, noul=2)"},
             "outputs": {"logits": "float32 [B,K] raw option logits (padded options = -1e4)", "act_logits": "float32 [B,2]"},
             "prompt": "[CLS] '<type> question: <instructions>' [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP] "
                       "(laya.common.build_sequence, max_len 1024, head_max_len 256); choice instructions = "
                       "'What does the user want to do?', options = 8 English descriptions; noul instructions = the question, "
                       "options = ['false: no, the statement does not hold', 'true: yes, the statement holds'], score = z_true - z_false"},
    "gli": {"inputs": {"input_ids": "int64 [B,S]", "attention_mask": "int64 [B,S]",
                       "label_slots": "int64 [K] (values ignored; its length K = number of labels)"},
            "outputs": {"logits": "float32 [B,K]"},
            "prompt": "'<<LABEL>>l1<<LABEL>>l2...<<SEP>>' + text, tokenizer adds [CLS]/[SEP]; choice = softmax over the 8 option "
                      "descriptions (single-label); noul = sigmoid of the single label = English intent description"},
}


def snippet(key, onnx_file):
    if key == "laya":
        return ('import sys; sys.path.insert(0, ".")   # run from inside the extracted folder\n'
                'import skejja_ortlib as ol\n'
                f'm = ol.LayaORT("{onnx_file}", "tokenizer.json", threads=4)\n'
                'p = ol.softmax(m.choice_logits("svegliami alle sette", ["set an alarm", "delete an alarm", "play music"]))\n'
                'p_yes = ol.sigmoid(m.noul_score("svegliami alle sette", "Does the user want to set an alarm?"))')
    return ('import sys; sys.path.insert(0, ".")   # run from inside the extracted folder\n'
            'import skejja_ortlib as ol\n'
            f'm = ol.GLiClassORT("{onnx_file}", "tokenizer.json", threads=4)\n'
            'p = ol.softmax(m.choice_logits("svegliami alle sette", ["set an alarm", "delete an alarm", "play music"]))\n'
            'p_yes = ol.sigmoid(m.noul_score("svegliami alle sette", "set an alarm"))')


def build(key, variant, onnx_path, metrics_csv, out_dir="out/bundles"):
    import onnxruntime, torch
    s = SPEC[key]
    q = pd.read_csv(os.path.join(WS, metrics_csv))
    sel = q[(q.model == s["name"]) & (q.variant == variant)]
    g = lambda task, metric, split="all", lang="all": float(sel[(sel.task == task) & (sel.metric == metric) & (sel.split == split) & (sel.lang == lang)].value.iloc[0])
    cal = {"choice_temperature": g("choice", "temperature"), "noul_platt_a": g("noul", "platt_a"), "noul_platt_b": g("noul", "platt_b")}
    met = {"choice_accuracy_macro": g("choice", "accuracy", lang="macro"), "choice_ece_cal_all": g("choice", "ece_cal"),
           "noul_auroc_macro": g("noul", "auroc", lang="macro"), "noul_ece_cal_all": g("noul", "ece_cal")}
    onnx_file = os.path.basename(onnx_path)
    info = {"model": s["name"], "source_repo": s["repo"], "source_revision": s["rev"], "license": "Apache-2.0",
            "variant": variant, "conversion": "torch.onnx.export(dynamo=True)" + QUANT[variant], "zero_shot": True,
            "onnx_file": onnx_file, "onnx_mb": os.path.getsize(os.path.join(WS, onnx_path)) / 1e6,
            "io": IO[key], "calibration_fitted_on": "Skejja calib_choice.jsonl / calib_noul.jsonl (1500 rows each, 5 langs pooled)",
            "calibration": cal, "test_metrics_mac_proxy": met,
            "tooling": {"torch": torch.__version__, "onnxruntime": onnxruntime.__version__}}
    readme = open(os.path.join(HERE, "BUNDLE_README.md")).read()
    rep = {"MODEL_NAME": s["name"], "VARIANT": variant, "REPO": s["repo"], "REV": s["rev"], "AUTHORS": s["authors"],
           "QUANT_NOTE": QUANT[variant], "ONNX_FILE": onnx_file, "TORCH": torch.__version__, "ORT": onnxruntime.__version__,
           "RUN_SNIPPET": snippet(key, onnx_file), "TEMP": "%.4f" % cal["choice_temperature"],
           "PLATT_A": "%.4f" % cal["noul_platt_a"], "PLATT_B": "%.4f" % cal["noul_platt_b"],
           "METRICS_NOTE": "\n".join("- %s: %.4f" % kv for kv in met.items())}
    for k, v in rep.items():
        readme = readme.replace("{" + k + "}", v)
    stage = os.path.join(WS, out_dir, "_stage_%s_%s" % (key, variant))
    os.makedirs(stage, exist_ok=True)
    open(os.path.join(stage, "model_info.json"), "w").write(json.dumps(info, indent=1))
    open(os.path.join(stage, "README.md"), "w").write(readme)
    open(os.path.join(stage, "NOTICE"), "w").write(
        "Derived from %s (revision %s) by %s, licensed under the Apache License 2.0 "
        "(https://www.apache.org/licenses/LICENSE-2.0). Converted to ONNX%s for Skejja phase-1 measurement; "
        "changes: format conversion%s only, weights otherwise unchanged.\n"
        % (s["repo"], s["rev"], s["authors"], " and quantized" if variant != "onnx-fp32" else "",
           "/quantization" if variant != "onnx-fp32" else ""))
    top = "skejja_baseline_%s_%s" % (s["name"], variant)
    tar_path = os.path.join(WS, out_dir, top + ".tar.gz")
    with tarfile.open(tar_path, "w:gz", compresslevel=6, dereference=True) as tf:
        tf.add(os.path.join(WS, onnx_path), arcname=f"{top}/{onnx_file}")
        for src, dst in s["tok_files"]:
            tf.add(os.path.join(s["base"], src), arcname=f"{top}/{dst}")
        tf.add(os.path.join(HERE, "skejja_ortlib.py"), arcname=f"{top}/skejja_ortlib.py")
        for f in ("model_info.json", "README.md", "NOTICE"):
            tf.add(os.path.join(stage, f), arcname=f"{top}/{f}")
    with tarfile.open(tar_path, "r:gz") as tf:
        names = tf.getnames()
    return tar_path, os.path.getsize(tar_path) / 1e6, names


if __name__ == "__main__":
    key, variant, onnx_path, metrics_csv = sys.argv[1:5]
    print(build(key, variant, onnx_path, metrics_csv))
