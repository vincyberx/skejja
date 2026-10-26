"""Child process for latency / peak-RSS measurement (one model x runtime x task x threads per process).

usage: python lat_child.py MODEL RUNTIME TASK THREADS MODE [ONNX_PATH]
  MODEL   laya | gli
  RUNTIME torch | ort
  TASK    choice | noul
  MODE    latency (10 warmup + 300 timed single decisions, prints LATENCY_JSON=...)
          rss     (load + 20 decisions, then frees the model and prints PEAK_RSS_MB=...; the high-water mark
                   is taken from getrusage so it includes the model; skejja_measure (scipy/sklearn) is imported only
                   AFTER the model is freed so its import cost does not inflate the number)
ORT runs never import torch.
"""
import gc, json, os, sys, time

MODEL, RUNTIME, TASK, THREADS, MODE = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
ONNX_PATH = sys.argv[6] if len(sys.argv) > 6 else None
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = str(THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["USE_TF"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "bench"))   # skejja_measure

import numpy as np

# dati: bench/data/ (identici byte per byte alle versioni dell'archivio di Claude Science usate in fase 1,
# verificato con SHA-256 il 16 ottobre 2026); SKEJJA_DATA sostituisce la cartella
DATA_DIR = os.environ.get("SKEJJA_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "data"))
TEST = {"choice": os.path.join(DATA_DIR, "test_choice.jsonl"),
        "noul": os.path.join(DATA_DIR, "test_noul.jsonl")}
DESC_PATH = os.path.join(DATA_DIR, "intent_descriptions.json")
rows = [json.loads(l) for l in open(TEST[TASK]) if l.strip()]
DESC = json.load(open(DESC_PATH))
rng = np.random.default_rng(0)
sel = [rows[i] for i in rng.permutation(len(rows))[:310]]   # fixed mixed-language sample, 10 warmup + 300 timed
N_RUN = 310 if MODE == "latency" else 20
sel = sel[:N_RUN]

WS = os.path.dirname(HERE)
_HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")   # come baseline_lib.HUB
LAYA_DIR = os.path.join(_HUB, "models--convaiinnovations--laya-multilingual/snapshots/1720e3e3357cfe1e281542e223f8273b0890ca34")
GLI_DIR = os.path.join(_HUB, "models--knowledgator--gliclass-multilang-edge/snapshots/d16c08ef70547514081952104e6fc3d190d9ee39")

if RUNTIME == "torch":
    import torch
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    import baseline_lib as bl
    runner = bl.LayaTorch() if MODEL == "laya" else bl.GliTorch()
    if TASK == "choice":
        fn = lambda r: runner.predict_choice(r["text"], r["option_desc"])
    else:
        fn = (lambda r: runner.predict_noul(r["text"], r["question"])) if MODEL == "laya" else \
             (lambda r: runner.predict_noul(r["text"], DESC[r["intent"]]))
    ntok = None
else:
    import skejja_ortlib as ol
    if MODEL == "laya":
        runner = ol.LayaORT(ONNX_PATH, os.path.join(LAYA_DIR, "tokenizer", "tokenizer.json"), threads=THREADS)
        fn = (lambda r: ol.softmax(runner.choice_logits(r["text"], r["option_desc"]))) if TASK == "choice" else \
             (lambda r: ol.sigmoid(runner.noul_score(r["text"], r["question"])))
    else:
        runner = ol.GLiClassORT(ONNX_PATH, os.path.join(GLI_DIR, "tokenizer.json"), threads=THREADS)
        fn = (lambda r: ol.softmax(runner.choice_logits(r["text"], r["option_desc"]))) if TASK == "choice" else \
             (lambda r: ol.sigmoid(runner.noul_score(r["text"], DESC[r["intent"]])))
    assert "torch" not in sys.modules, "torch was imported in an ORT child"

if MODE == "latency":
    import skejja_measure as fm
    res = fm.latency(fn, sel, warmup=10, n=300)
    res["torch_imported"] = "torch" in sys.modules
    print("LATENCY_JSON=" + json.dumps(res))
else:
    for r in sel:
        fn(r)
    torch_imported = "torch" in sys.modules
    del runner, fn
    gc.collect()
    import skejja_measure as fm
    print("TORCH_IMPORTED=%s" % torch_imported)
    print(f"PEAK_RSS_MB={fm.self_peak_rss_mb()}")
