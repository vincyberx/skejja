"""Skejja sui dataset dell'articolo arXiv 2609.37647 (valutazione indipendente di Jev), zero-shot.

Confronto con i numeri AGGREGATI pubblicati nell'articolo (Tabella 3): nessuna chiamata all'API di Jev
e nessun uso delle risposte di Jev pubblicate su Zenodo. Template/descrizioni delle opzioni scritti da noi
una sola volta, prima di vedere i risultati (l'articolo usa i propri template: non identici).

Uso:
  python bench/eval_jev_suite.py --data_dir <dir con i file scaricati> --bundles name=dir[:file.onnx] ... --out results/fase2/metrics_jev_suite.csv
  python bench/eval_jev_suite.py --download --data_dir <dir>     # scarica i file dal Hugging Face Hub
"""
import argparse, os, sys, time, json, csv
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runtime"))
from skejja_infer import Skejja  # noqa: E402

FILES = {  # nome -> (repo HF, file, revisione non fissata: annotata nel CSV come data di download)
    "agnews": ("fancyzhx/ag_news", "data/test-00000-of-00001.parquet"),
    "rotten": ("cornell-movie-review-data/rotten_tomatoes", "test.parquet"),
    "sst2": ("stanfordnlp/sst2", "data/validation-00000-of-00001.parquet"),
    "emotion": ("dair-ai/emotion", "split/test-00000-of-00001.parquet"),
    "finphrase": ("atrost/financial_phrasebank", "data/test-00000-of-00001-41c7ea948573445f.parquet"),
    "sms": ("ucirvine/sms_spam", "plain_text/train-00000-of-00001.parquet"),
    "langid": ("papluca/language-identification", "test.csv"),
    "banking77": ("mteb/banking77", "data/test-00000-of-00001.parquet"),
    **{f"sib_{l}": ("Davlan/sib200", f"data/{l}_Latn/test.tsv") for l in ["eng", "ita", "spa", "deu", "fra"]},
}

# Descrizioni delle opzioni (inglese, come in addestramento). Scritte una volta, non ritoccate dopo i risultati.
AG = ["World: international news, politics, conflicts and world events",
      "Sports: sports news, games, teams and athletes",
      "Business: business, economy, markets and companies",
      "Sci/Tech: science and technology news, computers, internet and research"]
SENT = ["negative: the review expresses a negative opinion of the movie",
        "positive: the review expresses a positive opinion of the movie"]
EMO = ["sadness", "joy", "love", "anger", "fear", "surprise"]
FIN = ["negative: bad news for the company or its investors",
       "neutral: neither good nor bad news for the company",
       "positive: good news for the company or its investors"]
LANG = {"ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
        "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
        "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai", "tr": "Turkish", "ur": "Urdu",
        "vi": "Vietnamese", "zh": "Chinese"}
SIB = {"science/technology": "science and technology", "travel": "travel and tourism", "politics": "politics",
       "sports": "sports", "health": "health and medicine", "entertainment": "entertainment",
       "geography": "geography and places"}
SPAM_Q = "receive spam: an unsolicited advertising, prize, scam or premium-rate message"

# Jev / Qwen3.8-27B / Gemma-4-E4B dalla Tabella 3 di arXiv 2609.37647 (metrica primaria, ECE).
PAPER = {"agnews": (0.885, 0.867, 0.862, 0.077), "rotten": (0.933, 0.932, 0.901, 0.038),
         "sst2": (0.964, 0.954, 0.952, 0.015), "emotion": (0.585, 0.571, 0.569, 0.279),
         "finphrase": (0.730, 0.677, 0.652, 0.138), "sms": (0.938, 0.940, 0.902, 0.033),
         "langid": (0.996, 0.998, 0.987, 0.003), "banking77": (0.797, 0.774, 0.670, 0.087),
         "sib200_all205": (0.815, 0.800, 0.761, 0.045)}


def load(data_dir):
    D = {}
    rd = lambda k, ext: os.path.join(data_dir, k + ext)
    d = pd.read_parquet(rd("agnews", ".parquet")); D["agnews"] = (d.text.tolist(), d.label.values, AG)
    d = pd.read_parquet(rd("rotten", ".parquet")); D["rotten"] = (d.text.tolist(), d.label.values, SENT)
    d = pd.read_parquet(rd("sst2", ".parquet")); D["sst2"] = (d.sentence.tolist(), d.label.values, SENT)
    d = pd.read_parquet(rd("emotion", ".parquet")); D["emotion"] = (d.text.tolist(), d.label.values, EMO)
    d = pd.read_parquet(rd("finphrase", ".parquet")); D["finphrase"] = (d.sentence.tolist(), d.label.values, FIN)
    d = pd.read_csv(rd("langid", ".csv")); codes = sorted(LANG)
    D["langid"] = (d.text.tolist(), np.array([codes.index(c) for c in d.labels]), [LANG[c] for c in codes])
    d = pd.read_parquet(rd("banking77", ".parquet")); names = sorted(set(d.label_text))
    D["banking77"] = (d.text.tolist(), np.array([names.index(x) for x in d.label_text]),
                      [n.replace("_", " ") for n in names])
    keys = list(SIB)
    for l in ["eng", "ita", "spa", "deu", "fra"]:
        d = pd.read_csv(rd(f"sib_{l}", ".tsv"), sep="\t")
        D[f"sib200_{l}"] = (d.text.tolist(), np.array([keys.index(c) for c in d.category]), [SIB[k] for k in keys])
    d = pd.read_parquet(rd("sms", ".parquet")); D["sms"] = (d.sms.tolist(), d.label.values, None)
    return D


def skejja_torch(bundle, ckpt, device, model_file=None):
    """Stessi pesi del bundle eseguiti da PyTorch (es. sulla GPU): tokenizer e calibrazione del bundle, `choice` e
    `noul` del runtime invariati; si sostituisce solo `logits`. Verificato su S1MB: |Δp| ≤ 6e-6 rispetto all'ONNX fp32."""
    import tokenizers
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
    import skejja_glic
    from gliclass import GLiClassModel
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scorer = skejja_glic.Scorer(GLiClassModel.from_pretrained(ckpt, dtype=torch.float32)).to(device).eval()
    cal = json.load(open(os.path.join(bundle, "calibration.json")))
    mf = model_file or sorted(k for k in cal if k.endswith(".fp32.onnx"))[0]
    sk = object.__new__(Skejja)
    sk.model_file, sk.cal, sk.sess = f"{mf}@torch", cal[mf], None
    sk.tok = tokenizers.Tokenizer.from_file(os.path.join(bundle, "tokenizer.json"))

    def logits(text, labels):
        s = "".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>" + text
        ids = torch.tensor([sk.tok.encode(s).ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            z = scorer(ids, torch.ones_like(ids), torch.zeros(len(labels), device=device))
        return z[0].float().cpu().numpy()

    sk.logits = logits
    return sk


def ece(p, y, bins=15):
    conf = p.max(1); pred = p.argmax(1); ok = pred == y; e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


def auroc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def boot(fn, n, B=1000, seed=0):
    rng = np.random.default_rng(seed); v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return np.percentile(v, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--bundles", nargs="*", default=[])
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="per i modelli name=bundle@checkpoint (es. cuda)")
    ap.add_argument("--out", default="results/fase2/metrics_jev_suite.csv")
    a = ap.parse_args()
    if a.download:
        from huggingface_hub import hf_hub_download
        import shutil
        os.makedirs(a.data_dir, exist_ok=True)
        for k, (r, f) in FILES.items():
            shutil.copy(hf_hub_download(r, f, repo_type="dataset"), os.path.join(a.data_dir, k + os.path.splitext(f)[1]))
        if not a.bundles: return
    D = load(a.data_dir)
    names = list(D) if a.datasets == "all" else a.datasets.split(",")
    rows = []
    raw = {}   # punteggi per esempio (per i confronti appaiati fra modelli): <modello>|<dataset> e y|<dataset>
    for spec in a.bundles:
        name, path = spec.split("=", 1); mf = None; ckpt = None
        if "@" in path: path, ckpt = path.split("@", 1)   # name=bundle@checkpoint: pesi PyTorch, --device
        if ":" in path: path, mf = path.split(":", 1)
        sk = skejja_torch(path, ckpt, a.device, mf) if ckpt else Skejja(path, threads=a.threads, model_file=mf)
        if sk.sess is not None and sk.sess.get_inputs()[2].type == "tensor(int64)":   # GLiClass originale: label_slots int64
            def _logits(text, labels, _sk=sk):
                s = "".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>" + text
                ids = np.array([_sk.tok.encode(s).ids], dtype=np.int64)
                return _sk.sess.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids),
                                                 "label_slots": np.zeros(len(labels), np.int64)})[0][0]
            sk.logits = _logits
        for ds in names:
            texts, y, opts = D[ds]; t0 = time.time()
            if opts is None:   # Noul (SMS spam), soglia 0,5 sulla probabilità calibrata (Platt globale)
                p = np.array([sk.noul(t, SPAM_Q) for t in texts])
                pred = (p >= 0.5).astype(int)
                def f1(ix):
                    tp = ((pred[ix] == 1) & (y[ix] == 1)).sum(); fp = ((pred[ix] == 1) & (y[ix] == 0)).sum()
                    fn = ((pred[ix] == 0) & (y[ix] == 1)).sum(); return 2 * tp / max(2 * tp + fp + fn, 1)
                n = len(y); lo, hi = boot(f1, n)
                P2 = np.stack([1 - p, p], 1)
                res = dict(metric="F1_pos", value=f1(np.arange(n)), ci_lo=lo, ci_hi=hi, ece=ece(P2, y),
                           auroc=auroc(p, y), acc=(pred == y).mean())
            else:
                P = np.stack([sk.choice(t, opts) for t in texts])
                ok = P.argmax(1) == y; n = len(y); lo, hi = boot(lambda ix: ok[ix].mean(), n)
                res = dict(metric="accuracy", value=ok.mean(), ci_lo=lo, ci_hi=hi, ece=ece(P, y), auroc="", acc=ok.mean())
            raw[f"{name}|{ds}"] = (p if opts is None else P).astype(np.float32); raw[f"y|{ds}"] = np.asarray(y)
            res.update(model=name, onnx=sk.model_file, dataset=ds, n=len(y), n_options=(len(opts) if opts else 1),
                       ms_per_item=1000 * (time.time() - t0) / len(y))
            rows.append(res)
            print(f"{name:10s} {ds:14s} {res['metric']} {res['value']:.3f} [{lo:.3f},{hi:.3f}] ece {res['ece']:.3f} "
                  f"{res['ms_per_item']:.1f} ms", flush=True)
    cols = ["model", "onnx", "dataset", "n", "n_options", "metric", "value", "ci_lo", "ci_hi", "ece", "auroc", "acc", "ms_per_item"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in cols})
    np.savez_compressed(os.path.splitext(a.out)[0] + "_scores.npz", **raw)
    with open(os.path.splitext(a.out)[0] + "_paper.json", "w") as fh:
        json.dump({"source": "arXiv 2609.37647, Table 3 (Jev, Qwen3.8-27B, Gemma-4-E4B, Jev ECE)",
                   "values": PAPER}, fh, indent=1)


if __name__ == "__main__":
    main()
