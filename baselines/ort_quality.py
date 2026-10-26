"""Compute raw scores with an ONNX variant on test+calib, ONE decision per session.run (batch=1, as deployed).
usage: python ort_quality.py MODEL(laya|gli) VARIANT ONNX_PATH THREADS OUT_NPZ
Batch=1 matters for int8: dynamic quantization picks the activation scale per tensor, so padding/batch
composition would otherwise change the numbers."""
import json, os, sys, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import skejja_ortlib as ol

MODEL, VARIANT, ONNX_PATH, THREADS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
# dati: bench/data/ (identici byte per byte alle versioni dell'archivio di Claude Science usate in fase 1,
# verificato con SHA-256 il 16 ottobre 2026); SKEJJA_DATA sostituisce la cartella
DATA_DIR = os.environ.get("SKEJJA_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "data"))
F = {"test_choice": os.path.join(DATA_DIR, "test_choice.jsonl"),
     "test_noul": os.path.join(DATA_DIR, "test_noul.jsonl"),
     "calib_choice": os.path.join(DATA_DIR, "calib_choice.jsonl"),
     "calib_noul": os.path.join(DATA_DIR, "calib_noul.jsonl")}
DESC = json.load(open(os.path.join(DATA_DIR, "intent_descriptions.json")))
WS = os.path.dirname(HERE)
LAYA_TOK = os.path.join(WS, ".cache/huggingface/hub/models--convaiinnovations--laya-multilingual/snapshots/1720e3e3357cfe1e281542e223f8273b0890ca34/tokenizer/tokenizer.json")
GLI_TOK = os.path.join(WS, ".cache/huggingface/hub/models--knowledgator--gliclass-multilang-edge/snapshots/d16c08ef70547514081952104e6fc3d190d9ee39/tokenizer.json")
runner = ol.LayaORT(ONNX_PATH, LAYA_TOK, threads=THREADS) if MODEL == "laya" else ol.GLiClassORT(ONNX_PATH, GLI_TOK, threads=THREADS)
out = {}
t0 = time.time()
for key, path in F.items():
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if key.endswith("choice"):
        out[key] = np.stack([runner.choice_logits(r["text"], r["option_desc"]) for r in rows])
    else:
        out[key] = np.array([runner.noul_score(r["text"], r["question"] if MODEL == "laya" else DESC[r["intent"]]) for r in rows])
    print(key, out[key].shape, round(time.time() - t0, 1), flush=True)
np.savez(OUT, **out)
print("DONE", OUT, round(time.time() - t0, 1))
