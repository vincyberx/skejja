"""Latenza e RAM di Julia 1 e di Skejja ft6 sulla stessa macchina, con le regole di skejja_measure.

Ogni misura gira in un processo nuovo (un modello x compito x thread x modo), come baselines/lat_child.py:
- latency: 10 decisioni di riscaldamento + 300 misurate, una decisione per chiamata (Choice a 8 opzioni o una sì/no),
  campione fisso misto per lingua (seed 0, lo stesso di lat_child.py);
- rss: caricamento + 20 decisioni, picco RSS del processo.
Julia 1: torch fp32 con il suo runtime (`engine.logits`, una riga per chiamata). ft6: runtime/skejja_infer.py su
onnxruntime, file fp32 e int8 del bundle. I numeri del Mac sono approssimazioni: valgono solo quelli misurati sul Pi.

uso: JULIA_DIR=<snapshot di Julia-1> python baselines/lat_julia.py --ft6 <bundle ft6 estratto> \
        --out results/fase3/latenza_julia-1_ft6_mac.json
"""
import argparse, json, os, platform, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SKEJJA_DATA", os.path.join(HERE, "..", "bench", "data"))
CHOICE_Q = "What does the user want to do?"


def child(model, task, threads, mode, path):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, f"test_{task}.jsonl")) if l.strip()]
    sel = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:310]]
    sel = sel[:310 if mode == "latency" else 20]
    if model == "julia-1":
        import torch
        torch.set_num_interop_threads(1)
        os.environ["JULIA_CPU_THREADS"] = str(threads)   # il runtime di julia ci rimette i thread al caricamento
        sys.path.insert(0, path)
        from julia import load_model
        eng = load_model(path, device="cpu", strict_encoding=True, max_length=1024, head_length=512)
        assert torch.get_num_threads() == threads, torch.get_num_threads()
        if task == "choice":
            fn = lambda r: eng.logits([dict(state=r["text"], question=CHOICE_Q, options=r["option_desc"], type="choice")])
        else:
            fn = lambda r: eng.logits([dict(state=r["text"], question=r["question"], options=["false", "true"], type="noul")])
    else:
        sys.path.insert(0, os.path.join(HERE, "..", "runtime"))
        from skejja_infer import Skejja
        sk = Skejja(path, threads=threads, model_file=f"edge-v32k-ft6.{model.split('-')[-1]}.onnx")
        fn = (lambda r: sk.choice(r["text"], r["option_desc"])) if task == "choice" else \
             (lambda r: sk.noul(r["text"], r["question"]))
    sys.path.insert(0, os.path.join(HERE, "..", "bench"))
    import skejja_measure as fm
    if mode == "latency":
        print("LATENCY_JSON=" + json.dumps(fm.latency(fn, sel, warmup=10, n=300)))
    else:
        for r in sel:
            fn(r)
        print(f"PEAK_RSS_MB={fm.self_peak_rss_mb()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--julia", default=os.environ.get("JULIA_DIR", ""))
    ap.add_argument("--ft6", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", default="1,4")
    ap.add_argument("--child", nargs=5, metavar=("MODEL", "TASK", "THREADS", "MODE", "PATH"))
    a = ap.parse_args()
    if a.child:
        m, t, th, mo, p = a.child
        return child(m, t, int(th), mo, p)
    res = dict(machine=dict(system=platform.system(), machine=platform.machine(), cpus=os.cpu_count(),
                            python=platform.python_version()), runs=[])
    for model, path in (("julia-1", a.julia), ("ft6-fp32", a.ft6), ("ft6-int8", a.ft6)):
        for task in ("choice", "noul"):
            for th in map(int, a.threads.split(",")):
                run = dict(model=model, task=task, threads=th)
                for mode in ("latency", "rss"):
                    out = subprocess.run([sys.executable, os.path.abspath(__file__), "--ft6", a.ft6, "--out", a.out,
                                          "--child", model, task, str(th), mode, path],
                                         check=True, capture_output=True, text=True).stdout
                    for l in out.splitlines():
                        if l.startswith("LATENCY_JSON="):
                            run.update(json.loads(l.split("=", 1)[1]))
                        elif l.startswith("PEAK_RSS_MB="):
                            run["peak_rss_mb"] = float(l.split("=", 1)[1])
                res["runs"].append(run)
                print(run, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
