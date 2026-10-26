"""Esperimento: calibrazione con pochi esempi per domanda (runtime.skejja_infer.fit_platt_map).

Per ogni domanda nuova si pescano k esempi etichettati (k/2 sì, k/2 no) e si stima una Platt MAP ancorata alla
calibrazione globale (Platt su calib_noul.jsonl, protocollo fase 1) con prior N(globale, sigma^2).
Gli esempi sono frasi distinte (si tiene la prima riga per id, cioè l'inglese); 'k=pool' = tutto il pool.
1. sigma si sceglie sul set di sviluppo esterno (dev_noul_ext.jsonl, 22 domande CLINC tenute fuori): esempi ed
   valutazione da frasi diverse della stessa domanda; criterio = ECE media con k = 16.
2. Test: per le 10 domande MASSIVE mai viste gli esempi vengono da calib_noul_fewshot.jsonl (MASSIVE validation), la
   misura da test_noul_unseen_ext.jsonl e test_noul.jsonl (split unseen_q); anche sulle sole frasi con etichetta 'ok'
   (bench/data/annotations/unseen_label_audit.csv). 30 estrazioni per k; si riportano media e percentili 5-95.
Richiede i logit salvati da eval_noul.py (metrics_<model>_scores.npz) e scores_fewshot_<model>.npz.

uso: python training/fewshot_calib.py --model edge32k-ft2 [--model edge32k-ft ...]
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

np.seterr(over="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bench")); sys.path.insert(0, os.path.join(HERE, "..", "runtime"))
import skejja_measure as fm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
# la funzione è presa dal sorgente del runtime (stesso codice distribuito), senza importare onnxruntime
_src = open(os.path.join(HERE, "..", "runtime", "skejja_infer.py")).read()
_ns = {"np": np}
exec(_src[_src.index("def fit_platt_map"):_src.index('if __name__ == "__main__":')], _ns)
fit_platt_map = _ns["fit_platt_map"]

SIGMAS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
KS = [4, 8, 16, 32, 64]
N_DRAWS = 30


def rd(p, **kw):
    d = pd.read_json(p, lines=True, dtype={"id": str}); return d


def draw(pool, k, rng):
    """k/2 positivi e k/2 negativi (o quanti ce ne sono), frasi distinte."""
    pos = pool[pool.label == 1]; neg = pool[pool.label == 0]
    ip = rng.choice(len(pos), size=min(k // 2, len(pos)), replace=False)
    ineg = rng.choice(len(neg), size=min(k - len(ip), len(neg)), replace=False)
    return pd.concat([pos.iloc[ip], neg.iloc[ineg]])


def ece_of(p, y):
    return fm.noul_metrics(p, y)["ece"]


def apply(df, params, prior):
    a = df.intent.map(lambda q: params.get(q, prior)[0]).values
    b = df.intent.map(lambda q: params.get(q, prior)[1]).values
    return fm.sigmoid(a * df.s.values + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "bench", "data"))
    ap.add_argument("--res", default=os.path.join(HERE, "..", "results", "fase2"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    D = a.data
    cal = rd(f"{D}/calib_noul.jsonl"); dev = rd(f"{D}/dev_noul_ext.jsonl"); fs = rd(f"{D}/calib_noul_fewshot.jsonl")
    ext = rd(f"{D}/test_noul_unseen_ext.jsonl"); te = rd(f"{D}/test_noul.jsonl")
    audit = pd.read_csv(f"{D}/annotations/unseen_label_audit.csv", dtype={"id": str}); code = dict(zip(audit.id, audit.code))
    rows = []
    for mn in a.model:
        z = np.load(f"{a.res}/metrics_{mn}_scores.npz"); zf = np.load(f"{a.res}/scores_fewshot_{mn}.npz")
        cal["s"] = z["calib"]; ext["s"] = z["ext"]; te["s"] = z["test"]; dev["s"] = zf["dev_ext"]; fs["s"] = zf["fewshot"]
        prior = fm.fit_platt(cal.s, cal.label)
        tu = te[te.intent.map(fm.is_unseen)].copy()
        ext_clean = ext[(ext.neg_type == "neg_other") | (ext.id.map(code) == "ok")]
        tu_c = tu.merge(pd.read_json(f"{D}/test_choice.jsonl", lines=True, dtype={"id": str})[["id", "lang", "gold"]],
                        on=["id", "lang"])
        tu_c["neg_type"] = np.where(tu_c.label == 1, "pos", np.where(tu_c.gold.str.split("_").str[0] == tu_c.intent.str.split("_").str[0], "neg_same", "neg_other"))
        tu_clean = tu_c[(tu_c.neg_type == "neg_other") | (tu_c.id.map(code) == "ok")]
        # ---- 1. scelta di sigma sullo sviluppo esterno
        rng = np.random.default_rng(a.seed)
        sel = {}
        for sg in SIGMAS:
            for k in KS:
                eces = []
                for _ in range(N_DRAWS):
                    params, held = {}, []
                    for q, pool in dev.groupby("intent"):
                        ex = draw(pool.drop_duplicates("id"), k, rng)
                        params[q] = fit_platt_map(ex.s, ex.label, prior, sg)
                        held.append(pool[~pool.id.isin(ex.id)])
                    h = pd.concat(held); eces.append(ece_of(apply(h, params, prior), h.label.values))
                sel[(sg, k)] = float(np.mean(eces))
                rows += fm.rows("fase2-fewshot", mn, f"map_sigma={sg}", "torch", 10, "noul", "dev_ext", f"k={k}",
                                {"ece_mean": sel[(sg, k)]}, len(dev))
        h = dev; rows += fm.rows("fase2-fewshot", mn, "global", "torch", 10, "noul", "dev_ext", "k=0",
                                 {"ece_mean": ece_of(fm.sigmoid(prior[0] * h.s + prior[1]), h.label.values)}, len(dev))
        best_sigma = min(SIGMAS, key=lambda sg: sel[(sg, 16)])
        rows += fm.rows("fase2-fewshot", mn, "selection", "torch", 10, "noul", "dev_ext", "k=16",
                        {"best_sigma": best_sigma, "platt_a_global": prior[0], "platt_b_global": prior[1]}, len(dev))
        # ---- 2. test MASSIVE mai viste
        evals = {"unseen_ext": ext, "unseen_ext_ok": ext_clean, "unseen_q": tu, "unseen_q_ok": tu_clean}
        for nm, d in evals.items():
            p = fm.sigmoid(prior[0] * d.s + prior[1])
            rows += fm.rows("fase2-fewshot", mn, "global", "torch", 10, "noul", nm, "k=0",
                            {"ece_mean": ece_of(p, d.label.values), "auroc_mean": roc_auc_score(d.label, d.s),
                             "accuracy_mean": float(((p >= 0.5) == d.label).mean())}, len(d))
        for k in KS:
            res = {nm: {"ece": [], "auroc": [], "acc": []} for nm in evals}
            for _ in range(N_DRAWS):
                params = {}
                for q, pool in fs.groupby("intent"):
                    ex = draw(pool.drop_duplicates("id"), k, rng)
                    params[q] = fit_platt_map(ex.s, ex.label, prior, best_sigma)
                for nm, d in evals.items():
                    p = apply(d, params, prior)
                    res[nm]["ece"].append(ece_of(p, d.label.values)); res[nm]["auroc"].append(roc_auc_score(d.label, p))
                    res[nm]["acc"].append(float(((p >= 0.5) == d.label).mean()))
            for nm in evals:
                m = {}
                for key, v in res[nm].items():
                    m[f"{key}_mean"] = float(np.mean(v)); m[f"{key}_p05"] = float(np.percentile(v, 5)); m[f"{key}_p95"] = float(np.percentile(v, 95))
                rows += fm.rows("fase2-fewshot", mn, f"map_sigma={best_sigma}", "torch", 10, "noul", nm, f"k={k}", m, len(evals[nm]))
        # pool completo (tutte le frasi e le lingue del pool per domanda): limite del metodo con molti esempi
        for sg in (best_sigma, 1e3):
            params = {q: fit_platt_map(pool.s, pool.label, prior, sg) for q, pool in fs.groupby("intent")}
            for nm, d in evals.items():
                p = apply(d, params, prior)
                rows += fm.rows("fase2-fewshot", mn, f"map_sigma={sg}", "torch", 10, "noul", nm, "k=pool",
                                {"ece_mean": ece_of(p, d.label.values), "auroc_mean": roc_auc_score(d.label, p),
                                 "accuracy_mean": float(((p >= 0.5) == d.label).mean())}, len(d))
        print(mn, "sigma scelto:", best_sigma, flush=True)
    out = pd.DataFrame(rows, columns=fm.METRIC_COLUMNS)
    out.to_csv(f"{a.res}/metrics_fewshot.csv", index=False)
    t = out[(out.split != "dev_ext") & out.metric.isin(["ece_mean", "auroc_mean", "ece_p95"])]
    print(t.pivot_table(index=["model", "split", "lang"], columns="metric", values="value").round(3).to_string())


if __name__ == "__main__":
    main()
