"""Skejja phase 1, Baseline track: torch runners (zero-shot) for Laya multilingual and GLiClass multilang edge."""
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# cache dei modelli: quella standard di Hugging Face (HF_HUB_CACHE, poi HF_HOME/hub), non più una .cache dentro il
# workspace di fase 1; revisioni fissate. `python baselines/baseline_lib.py --scarica` le scarica se mancano.
HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")
LAYA_REPO, LAYA_REV = "convaiinnovations/laya-multilingual", "1720e3e3357cfe1e281542e223f8273b0890ca34"
GLI_REPO, GLI_REV = "knowledgator/gliclass-multilang-edge", "d16c08ef70547514081952104e6fc3d190d9ee39"
LAYA_DIR = os.path.join(HUB, "models--convaiinnovations--laya-multilingual", "snapshots", LAYA_REV)
GLI_DIR = os.path.join(HUB, "models--knowledgator--gliclass-multilang-edge", "snapshots", GLI_REV)

# dati: bench/data/ (identici byte per byte alle versioni dell'archivio di Claude Science usate in fase 1,
# verificato con SHA-256 il 16 ottobre 2026); SKEJJA_DATA sostituisce la cartella
DATA_DIR = os.environ.get("SKEJJA_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "data"))
DATA = {
    "test_choice": os.path.join(DATA_DIR, "test_choice.jsonl"),
    "test_noul": os.path.join(DATA_DIR, "test_noul.jsonl"),
    "calib_choice": os.path.join(DATA_DIR, "calib_choice.jsonl"),
    "calib_noul": os.path.join(DATA_DIR, "calib_noul.jsonl"),
    "intent_desc": os.path.join(DATA_DIR, "intent_descriptions.json"),
}

LAYA_CHOICE_INS = "What does the user want to do?"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def intent_desc():
    with open(DATA["intent_desc"]) as f:
        return json.load(f)


def noul_label(row, desc=None):
    """GLiClass noul label = English intent description (== the phrase inside the question)."""
    desc = desc or intent_desc()
    return desc[row["intent"]]


# ---------------- Laya (torch) ----------------
class LayaTorch:
    def __init__(self):
        import laya
        self.agent = laya.load(LAYA_DIR, device="cpu")
        self.tok = self.agent.tok

    @staticmethod
    def choice_q(options):
        return {"intent": {"type": "choice", "instructions": LAYA_CHOICE_INS, "criteria": list(options)}}

    @staticmethod
    def noul_q(question):
        return {"q": {"type": "noul", "instructions": question}}

    def encode(self, text, qdict):
        a = self.agent
        ids = list(qdict)
        for k in ids:
            a._check_question(k, qdict[k])
        internal = {k: a._to_internal(v) for k, v in qdict.items()}
        return a._encode_state(text, ids, internal)

    def raw_logits(self, texts, qdicts, bs=32):
        """Same encode -> collate -> forward path as Agent.predict_batch, returning the raw (un-tempered) option logits."""
        import torch
        from laya.common import collate_items
        order = np.argsort([len(t) for t in texts])
        out = [None] * len(texts)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            items = [self.encode(texts[i], qdicts[i]) for i in idx]
            b = collate_items(items, self.tok.pad_token_id)
            with torch.no_grad():
                logits, _act = self.agent._forward(b)
            for j, i in enumerate(idx):
                k = len(items[j][0]["markers"])
                out[i] = logits[j, :k].astype(np.float64)
        return out

    def predict_choice(self, text, options):          # public API, one decision
        return self.agent.predict(text, self.choice_q(options))

    def predict_noul(self, text, question):
        return self.agent.predict(text, self.noul_q(question))


# ---------------- GLiClass (torch) ----------------
class GliTorch:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        self.model = GLiClassModel.from_pretrained(GLI_DIR, dtype=torch.float32).eval()
        self.tok = AutoTokenizer.from_pretrained(GLI_DIR)
        self.pipe_single = ZeroShotClassificationPipeline(self.model, self.tok, classification_type="single-label",
                                                          device="cpu", progress_bar=False)
        self.pipe_multi = ZeroShotClassificationPipeline(self.model, self.tok, classification_type="multi-label",
                                                         device="cpu", progress_bar=False)

    def raw_logits(self, texts, label_lists, bs=32):
        """Same prepare_inputs -> model path as ZeroShotClassificationPipeline.__call__, returning raw logits."""
        import torch
        inner = self.pipe_single.pipe
        order = np.argsort([len(t) for t in texts])
        out = [None] * len(texts)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            tb = [texts[i] for i in idx]
            lb = [list(label_lists[i]) for i in idx]
            tokd = inner.prepare_inputs(tb, lb, same_labels=False)
            mnc = inner._resolve_max_num_classes(lb, False)
            with torch.no_grad():
                logits = self.model(**tokd, max_num_classes=mnc, adapter_ids=None).logits
            logits = logits.float().numpy()
            for j, i in enumerate(idx):
                out[i] = logits[j, :len(lb[j])].astype(np.float64)
        return out

    def predict_choice(self, text, options):          # public API, one decision
        return self.pipe_single(text, list(options), classification_type="single-label")

    def predict_noul(self, text, label):
        return self.pipe_multi(text, [label], classification_type="multi-label", threshold=0.5)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Percorsi della traccia baseline; --scarica scarica i due modelli alle "
                                             "revisioni fissate nella cache di Hugging Face")
    ap.add_argument("--scarica", action="store_true")
    a = ap.parse_args()
    if a.scarica:
        from huggingface_hub import snapshot_download
        for repo, rev in ((LAYA_REPO, LAYA_REV), (GLI_REPO, GLI_REV)):
            print(snapshot_download(repo, revision=rev, cache_dir=HUB))
    for k, v in dict(HUB=HUB, LAYA_DIR=LAYA_DIR, GLI_DIR=GLI_DIR, **DATA).items():
        print(f"{k:<14}{'ok ' if os.path.exists(v) else 'MANCA'} {v}")
