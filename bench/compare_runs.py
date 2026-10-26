"""Differenze appaiate fra due modelli su MASSIVE (stessi item), con intervalli bootstrap al 95% per frase.

Legge i punteggi salvati da training/eval_noul.py (`<metrics>_scores.npz`: `choice_test` (N,8), `test`, `ext`) e
calcola, per B − A: accuratezza Choice su tutte le intenzioni e su quelle mai viste; AUROC sì/no sul test originale e
sul test esteso delle intenzioni mai viste; se i file hanno `choice_test_native`, anche la Choice con le opzioni nella
lingua della frase. Il bootstrap ricampiona le frasi (le 5 lingue di una frase restano insieme), 10.000 ripetizioni
(`--B`; fino al 16 ottobre il valore era fisso a 1.000, contro i 10.000 dichiarati nella regola dei giri G e H).

uso: python bench/compare_runs.py --a results/fase2/metrics_edge32k-ft5_scores.npz --name_a edge32k-ft5 \
        --b results/fase2/metrics_edge32k-ft6_scores.npz --name_b edge32k-ft6 --out results/fase2/metrics_diff_ft6_vs_ft5.csv
"""
import argparse, csv, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skejja_measure as fm  # noqa: E402


def read(p):
    return [json.loads(l) for l in open(p)]


def boot(fn, groups, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ix = np.concatenate([idx[g] for g in pick])
        vals.append(fn(ix))
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--name_a", required=True)
    ap.add_argument("--b", required=True); ap.add_argument("--name_b", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10000, help="ricampionamenti del bootstrap")
    a = ap.parse_args()
    A, Bz = np.load(a.a), np.load(a.b)
    ch = read(os.path.join(a.data, "test_choice.jsonl"))
    y = np.array([r["gold_idx"] for r in ch]); g = np.array([r["id"] for r in ch])
    unseen = np.array([fm.is_unseen(r["gold"]) for r in ch])
    okA = A["choice_test"].argmax(1) == y; okB = Bz["choice_test"].argmax(1) == y
    righe = []

    def add(split, metric, val, lo, hi, n):
        righe.append(dict(confronto=f"{a.name_b} - {a.name_a}", split=split, metrica=metric, valore=round(float(val), 4),
                          ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4), n=int(n)))

    casi = [("choice/tutte", np.ones_like(unseen), okA, okB), ("choice/mai_viste", unseen, okA, okB)]
    if "choice_test_native" in A.files and "choice_test_native" in Bz.files:
        casi.append(("choice/opzioni_nella_lingua", np.ones_like(unseen),
                     A["choice_test_native"].argmax(1) == y, Bz["choice_test_native"].argmax(1) == y))
    for split, m, okA, okB in casi:
        ix0 = np.where(m)[0]
        d = lambda ix: okB[ix0[ix]].mean() - okA[ix0[ix]].mean()
        lo, hi = boot(d, g[ix0], B=a.B)
        add(split, "diff_accuratezza", d(np.arange(len(ix0))), lo, hi, len(ix0))
        add(split, f"accuratezza_{a.name_a}", okA[ix0].mean(), np.nan, np.nan, len(ix0))
        add(split, f"accuratezza_{a.name_b}", okB[ix0].mean(), np.nan, np.nan, len(ix0))
    for key, fname, split in (("test", "test_noul.jsonl", "noul/test"), ("ext", "test_noul_unseen_ext.jsonl", "noul/esteso_mai_viste")):
        rows = read(os.path.join(a.data, fname))
        yy = np.array([r["label"] for r in rows]); gg = np.array([str(r["id"]) for r in rows])
        sA, sB = A[key], Bz[key]
        d = lambda ix: roc_auc_score(yy[ix], sB[ix]) - roc_auc_score(yy[ix], sA[ix])
        lo, hi = boot(d, gg, B=a.B)
        add(split, "diff_auroc", d(np.arange(len(yy))), lo, hi, len(yy))
        add(split, f"auroc_{a.name_a}", roc_auc_score(yy, sA), np.nan, np.nan, len(yy))
        add(split, f"auroc_{a.name_b}", roc_auc_score(yy, sB), np.nan, np.nan, len(yy))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0])); w.writeheader(); w.writerows(righe)
    for r in righe:
        print(f"{r['split']:<24}{r['metrica']:<28}{r['valore']:>8}  [{r['ci_lo']}, {r['ci_hi']}]  n={r['n']}")


if __name__ == "__main__":
    main()
