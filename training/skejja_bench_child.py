"""Fresh-process benchmark child for Skejja small models.
mode=latency : skejja_measure.latency over ONE decision per call (tokenization + forward), warmup 10, n timed calls.
mode=rss     : load model, run 20 decisions (10 choice + 10 noul), print PEAK_RSS_MB (skejja_measure.self_peak_rss_mb).
Tokenization uses the `tokenizers` library on tokenizer.json (same ids as transformers; verified), as on the Pi.
usage: python skejja_bench_child.py <wd> <mode> <runtime torch|onnxruntime> <model_path> <tok_dir> <threads> <task> <n> <inputs.json>
"""
import json, os, sys

wd, mode, runtime, path, tok_dir, threads, task, n, inputs_path = sys.argv[1:10]
threads = int(threads); n = int(n)
sys.path.insert(0, wd)
os.environ["OMP_NUM_THREADS"] = str(threads)
import numpy as np
import resource
LEAN = os.environ.get("SKEJJA_LEAN") == "1" and mode == "rss"
if not LEAN:
    import skejja_measure as fm   # pulls scipy + sklearn (~150 MB RSS) -- protocol-faithful child
import tokenizers


def _peak_mb():  # identical formula to skejja_measure.self_peak_rss_mb (inlined so the lean child avoids scipy/sklearn)
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(r / 1e6 if sys.platform == "darwin" else r / 1024)

tk = tokenizers.Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
data = json.load(open(inputs_path))  # {"choice": [str], "noul": [str]}

if runtime == "torch":
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from gliclass import GLiClassModel
    import skejja_glic as fg
    model = fg.Scorer(GLiClassModel.from_pretrained(path, dtype=torch.float32).eval()).eval()
    slots = {8: torch.zeros(8), 1: torch.zeros(1)}

    def decide(s, K):
        ids = tk.encode(s).ids
        with torch.inference_mode():
            x = torch.tensor([ids]); m = torch.ones_like(x)
            return model(x, m, slots[K]).numpy()
else:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if os.environ.get("SKEJJA_ORT_LOWMEM") == "1":
        # low-memory session: no weight pre-packing copy, no CPU memory arena, no memory-pattern planning
        so.add_session_config_entry("session.disable_prepacking", "1")
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    slots = {8: np.zeros(8, np.float32), 1: np.zeros(1, np.float32)}

    def decide(s, K):
        ids = np.array([tk.encode(s).ids], dtype=np.int64)
        return sess.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids), "label_slots": slots[K]})[0]

if mode == "latency":
    K = 8 if task == "choice" else 1
    xs = data[task][: 10 + n]
    r = fm.latency(lambda s: decide(s, K), xs, warmup=10, n=n)
    print("RESULT=" + json.dumps(r))
else:
    for s in data["choice"][:10]:
        decide(s, 8)
    for s in data["noul"][:10]:
        decide(s, 1)
    print(f"PEAK_RSS_MB={_peak_mb() if LEAN else fm.self_peak_rss_mb()}")
