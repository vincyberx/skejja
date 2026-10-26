"""Shared measurement helpers for Skejja phase 1 (both tracks must use these so numbers are comparable).

Conventions
- Choice: each item has 8 options; model returns a vector of 8 scores (logits or probabilities).
- Noul: each item has one yes/no question; model returns one score (logit or probability of "yes").
- Calibration (temperature for Choice, Platt for Noul) is fitted ONLY on calib_* files, evaluated on test_*.
- Latency: one decision = one state + one question (Choice with 8 options, or one Noul) in a single call.
  Measure with CPU threads fixed (4 = Raspberry-Pi-like core count, 1 = worst case), after warmup.
- Memory: peak RSS of a FRESH child process that loads the model and runs a few inferences.
- Unseen split: items whose gold intent belongs to HELD_OUT_SCENARIOS are 'unseen' (never used in any training).
"""
import json, math, os, subprocess, sys, time, resource
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

LANGS = ["en", "it", "es", "de", "fr"]
HELD_OUT_SCENARIOS = ["transport", "takeaway", "social", "cooking"]
METRIC_COLUMNS = ["track", "model", "variant", "runtime", "threads", "task", "split", "lang", "metric", "value", "n"]


def scenario(intent):
    return intent.split("_")[0]


def is_unseen(intent):
    return scenario(intent) in HELD_OUT_SCENARIOS


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def ece(conf, correct, bins=15):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


# ---------- Choice ----------
def choice_metrics(probs, y):
    """probs: (N,K) probabilities; y: (N,) gold index."""
    probs = np.asarray(probs, float); y = np.asarray(y, int)
    pred = probs.argmax(1); conf = probs.max(1); correct = (pred == y)
    onehot = np.eye(probs.shape[1])[y]
    return {
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "ece": ece(conf, correct),
    }


def fit_temperature(logits, y):
    """logits: (N,K) raw scores (use log-probs if the model only gives probabilities)."""
    logits = np.asarray(logits, float); y = np.asarray(y, int)
    def nll(logT):
        p = softmax(logits, math.exp(logT))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    r = minimize_scalar(nll, bounds=(-4, 4), method="bounded")
    return float(math.exp(r.x))


# ---------- Noul ----------
def noul_metrics(p_yes, y):
    p = np.asarray(p_yes, float); y = np.asarray(y, int)
    pred = (p >= 0.5).astype(int); conf = np.where(pred == 1, p, 1 - p)
    return {
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "accuracy": float((pred == y).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "nll": float(-(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))).mean()),
        "ece": ece(conf, pred == y),
    }


def fit_platt(scores, y):
    """scores: raw logits (or logit(p)). Returns (a, b) so that p = sigmoid(a*s + b)."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(scores, float).reshape(-1, 1), np.asarray(y, int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


# ---------- Latency / memory ----------
def latency(fn, inputs, warmup=10, n=300):
    """fn(x) runs ONE decision. Returns dict p50_ms, p95_ms, mean_ms, n."""
    for x in inputs[:warmup]:
        fn(x)
    ts = []
    for x in inputs[warmup:warmup + n]:
        t0 = time.perf_counter(); fn(x); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "mean_ms": float(ts.mean()), "n": int(len(ts))}


def self_peak_rss_mb():
    """Peak RSS of the CURRENT process in MB (macOS reports bytes, Linux kilobytes)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(r / 1e6 if sys.platform == "darwin" else r / 1024)


def peak_rss_mb(script_path, *args, env=None):
    """Run `python script_path args` in a fresh process. The child script must end with
    print(f"PEAK_RSS_MB={self_peak_rss_mb()}"). Returns that value."""
    out = subprocess.run([sys.executable, script_path, *map(str, args)], check=True, env=env,
                         capture_output=True, text=True).stdout
    vals = [float(l.split("=", 1)[1]) for l in out.splitlines() if l.startswith("PEAK_RSS_MB=")]
    if not vals:
        raise RuntimeError("child did not print PEAK_RSS_MB=")
    return vals[-1]


def rows(track, model, variant, runtime, threads, task, split, lang, metrics, n):
    return [dict(track=track, model=model, variant=variant, runtime=runtime, threads=threads, task=task,
                 split=split, lang=lang, metric=k, value=v, n=n) for k, v in metrics.items()]


# ---------- Fase 2: Noul per tipo di negativo, bootstrap per frase ----------
def noul_neg_type(rows, choice_rows):
    """Tipo di ogni item Noul: 'pos', 'neg_same' (negativo dello stesso ambito) o 'neg_other'.
    In test_noul/calib_noul il campo `intent` è l'intenzione della DOMANDA; l'intenzione vera della frase si prende
    dal file Choice corrispondente (stessi id e lingue). I file che hanno già `gold`/`neg_type` li usano."""
    gold = {(r["id"], r["lang"]): r["gold"] for r in choice_rows} if choice_rows else {}
    out = []
    for r in rows:
        if "neg_type" in r:
            out.append(r["neg_type"]); continue
        g = r.get("gold") or gold[(r["id"], r["lang"])]
        out.append("pos" if r["label"] == 1 else ("neg_same" if scenario(g) == scenario(r["intent"]) else "neg_other"))
    return out


def bootstrap_ci(fn, groups, n=1000, seed=0, alpha=0.05):
    """Intervallo bootstrap (percentili) ricampionando i GRUPPI (es. id della frase: le 5 lingue restano insieme).
    fn(idx) -> metrica sugli indici idx. Restituisce (lo, hi)."""
    groups = np.asarray(groups); uniq = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        idx = np.concatenate([pos[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        try:
            v = fn(idx)
        except ValueError:
            continue
        if not np.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def open_mt_cache(ext_dir):
    """Cache delle traduzioni OPUS-MT: mt_cache.jsonl se presente (in lavorazione), altrimenti mt_cache.jsonl.gz
    (versione in git)."""
    import gzip
    p = os.path.join(ext_dir, "mt_cache.jsonl")
    return open(p) if os.path.exists(p) else gzip.open(p + ".gz", "rt")
