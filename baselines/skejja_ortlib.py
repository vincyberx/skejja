"""Torch-free inference helpers for the Skejja baseline ONNX exports (numpy + tokenizers + onnxruntime only).

Two runners, both exposing
    choice_logits(text, options)  -> np.ndarray (K,)   raw option logits (softmax -> probabilities)
    noul_score(text, question_or_label) -> float       raw logit of "yes" (sigmoid -> probability)

LayaORT     : Laya multilingual (convaiinnovations/laya-multilingual) exported DecisionModel.
              Re-implements laya.common.build_head/build_sequence (laya 0.3.26) for plain-string states.
              ONNX inputs: input_ids, attention_mask, marker_pos, marker_mask(bool), qtype -> logits, act_logits
              (same I/O names as laya.onnx_agent.ONNXAgent, so the file also works with laya's own ONNX path).
GLiClassORT : knowledgator/gliclass-multilang-edge uni-encoder.
              Input text = "<<LABEL>>l1<<LABEL>>l2...<<SEP>>" + text (prompt_first=True), [CLS]...[SEP] added.
              ONNX inputs: input_ids, attention_mask, label_slots (int64, shape (K,), only its length is used)
              -> logits (B, K).

No torch import anywhere in this file (so it can run on a Raspberry Pi with onnxruntime only).
"""
import json
import os

import numpy as np


def make_session(onnx_path, threads=4, opt_level="all"):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    }[opt_level]
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


def _load_tokenizer(tok_json):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_json)
    tok.no_truncation()
    tok.no_padding()
    return tok


# ----------------------------------------------------------------------------------------------
class LayaORT:
    QTYPES = {"choice": 0, "score": 1, "noul": 2}
    NOUL_OPTIONS = ["false: no, the statement does not hold", "true: yes, the statement holds"]
    CHOICE_INSTRUCTIONS = "What does the user want to do?"

    def __init__(self, onnx_path, tokenizer_json, threads=4, max_len=1024, head_max_len=256, session=None):
        self.tok = _load_tokenizer(tokenizer_json)
        self.max_len, self.head_max_len = max_len, head_max_len
        self.mask_tok = "<mask>"
        tid = self.tok.token_to_id
        self.cls_id, self.sep_id, self.mask_id, self.pad_id = tid("<bos>"), tid("<eos>"), tid("<mask>"), tid("<pad>")
        assert None not in (self.cls_id, self.sep_id, self.mask_id, self.pad_id)
        self.sess = session if session is not None else make_session(onnx_path, threads)

    def _enc(self, text, max_length=None):
        ids = self.tok.encode(text, add_special_tokens=False).ids
        return ids[:max_length] if max_length else ids

    def build(self, state, qtype, instructions, options):
        """Mirror of laya.common.build_sequence for a plain-string state (no option_order, right truncation)."""
        m = self.mask_tok
        head_ids = self._enc("%s question: %s" % (qtype, str(instructions).replace(m, " ")))
        opt_ids = [[self.mask_id] + self._enc(" " + o.replace(m, " "), 48) for o in options]
        opt_budget = self.head_max_len - sum(len(o) for o in opt_ids)
        if opt_budget < 16:
            per = max(4, (self.head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = self.head_max_len - sum(len(o) for o in opt_ids)
        head_ids = head_ids[: max(8, opt_budget)]
        ids = [self.cls_id] + head_ids + [self.sep_id]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(self.sep_id)
        room = max(0, self.max_len - len(ids) - 1)
        st = self._enc(state.replace(m, " "))[:room]
        ids = ids + st + [self.sep_id]
        ids, markers = ids[: self.max_len], [x for x in markers if x < self.max_len]
        return ids, markers

    def _run(self, rows, qtype):
        n, L = len(rows), max(len(r[0]) for r in rows)
        kmax = max(len(r[1]) for r in rows)
        ids = np.full((n, L), self.pad_id, dtype=np.int64)
        att = np.zeros((n, L), dtype=np.int64)
        mpos = np.zeros((n, kmax), dtype=np.int64)
        mmask = np.zeros((n, kmax), dtype=bool)
        for i, (s, mk) in enumerate(rows):
            ids[i, : len(s)] = s
            att[i, : len(s)] = 1
            mpos[i, : len(mk)] = mk
            mmask[i, : len(mk)] = True
        qt = np.full((n,), self.QTYPES[qtype], dtype=np.int64)
        logits, _act = self.sess.run(["logits", "act_logits"], {
            "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "qtype": qt})
        return [logits[i, : len(r[1])].astype(np.float64) for i, r in enumerate(rows)]

    def choice_logits(self, text, options, instructions=None):
        row = self.build(text, "choice", instructions or self.CHOICE_INSTRUCTIONS, list(options))
        return self._run([row], "choice")[0]

    def noul_score(self, text, question):
        row = self.build(text, "noul", question, self.NOUL_OPTIONS)
        z = self._run([row], "noul")[0]
        return float(z[1] - z[0])            # logit of "true" vs "false" (sigmoid -> p_yes)

    def choice_logits_batch(self, texts, options_list, instructions=None):
        rows = [self.build(t, "choice", instructions or self.CHOICE_INSTRUCTIONS, list(o)) for t, o in zip(texts, options_list)]
        return self._run(rows, "choice")

    def noul_score_batch(self, texts, questions):
        rows = [self.build(t, "noul", q, self.NOUL_OPTIONS) for t, q in zip(texts, questions)]
        return [float(z[1] - z[0]) for z in self._run(rows, "noul")]


# ----------------------------------------------------------------------------------------------
class GLiClassORT:
    def __init__(self, onnx_path, tokenizer_json, threads=4, session=None, max_length=1024):
        self.tok = _load_tokenizer(tokenizer_json)
        self.pad_id = 0
        self.max_length = max_length
        self.sess = session if session is not None else make_session(onnx_path, threads)

    @staticmethod
    def prompt(text, labels):
        return "".join("<<LABEL>>" + l for l in labels) + "<<SEP>>" + text

    def _run(self, texts, label_lists):
        encs = [self.tok.encode(self.prompt(t, ls), add_special_tokens=True).ids[: self.max_length]
                for t, ls in zip(texts, label_lists)]
        n, L = len(encs), max(len(e) for e in encs)
        K = max(len(ls) for ls in label_lists)
        ids = np.full((n, L), self.pad_id, dtype=np.int64)
        att = np.zeros((n, L), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, : len(e)] = e
            att[i, : len(e)] = 1
        logits = self.sess.run(["logits"], {"input_ids": ids, "attention_mask": att,
                                            "label_slots": np.zeros((K,), dtype=np.int64)})[0]
        return [logits[i, : len(ls)].astype(np.float64) for i, ls in enumerate(label_lists)]

    def choice_logits(self, text, options):
        return self._run([text], [list(options)])[0]

    def noul_score(self, text, label):
        return float(self._run([text], [[label]])[0][0])

    def choice_logits_batch(self, texts, options_list):
        return self._run(texts, [list(o) for o in options_list])

    def noul_score_batch(self, texts, labels):
        return [float(z[0]) for z in self._run(texts, [[l] for l in labels])]


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
