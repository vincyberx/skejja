"""ONNX export (fp32) + dynamic int8 quantization for the Skejja GLiClass Scorer, and ORT scoring helpers."""
import os, warnings
import numpy as np
import torch


def export_onnx(scorer, tok, path, opset=17):
    scorer.eval()
    enc = tok(["<<LABEL>>turn on the lights<<LABEL>>play music<<SEP>>switch the lamp on please"], return_tensors="pt")
    args = (enc["input_ids"], enc["attention_mask"], torch.zeros(2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(scorer, args, path, input_names=["input_ids", "attention_mask", "label_slots"],
                          output_names=["logits"], opset_version=opset, dynamo=False,
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                        "label_slots": {0: "num_labels"}, "logits": {0: "batch", 1: "num_labels"}})
    return path


def quantize_int8(src, dst, per_channel=True, exclude_patterns=()):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    nodes_to_exclude = []
    if exclude_patterns:
        m = onnx.load(src)
        nodes_to_exclude = [n.name for n in m.graph.node if any(p in n.name for p in exclude_patterns)]
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=per_channel,
                     op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=nodes_to_exclude)
    return dst, len(nodes_to_exclude)


def ort_session(path, threads=4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def score_ort(sess, tok, inputs, K, batch_size=64, max_length=512):
    order = np.argsort([len(s) for s in inputs]); inv = np.empty_like(order); inv[order] = np.arange(len(order))
    srt = [inputs[i] for i in order]; out = []
    slots = np.zeros(K, dtype=np.float32)
    for i in range(0, len(srt), batch_size):
        enc = tok(srt[i:i + batch_size], truncation=True, max_length=max_length, padding="longest", return_tensors="np")
        lg = sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                    "attention_mask": enc["attention_mask"].astype(np.int64), "label_slots": slots})[0]
        out.append(lg[:, :K].astype(np.float32))
    return np.concatenate(out)[inv]
