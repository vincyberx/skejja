"""Valutazione sul test fuori da MASSIVE (bench/data/ood/ood_v0.jsonl): opzioni e domande in italiano e in inglese.

Calibrazione: solo quella del protocollo (temperatura su calib_choice.jsonl, Platt su calib_noul.jsonl), cioè il caso
"domanda nuova senza esempi". Intervalli bootstrap per id (frase/azione/email).

uso: python training/eval_ood.py --model releases/checkpoints/edge32k-ft --name edge32k-ft --out results/fase2/metrics_ood_v0.csv
     (più --model/--name ripetuti per più modelli)
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "bench"))
import skejja_measure as fm  # noqa: E402
import skejja_glic as fg  # noqa: E402
from eval_noul import Backend, read  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

np.seterr(over="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "bench", "data"))
    ap.add_argument("--file", default="ood/ood_v0.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    D = a.data
    items = pd.DataFrame(read(f"{D}/{a.file}"))
    cc, cn = read(f"{D}/calib_choice.jsonl"), read(f"{D}/calib_noul.jsonl")
    rows, raw = [], {}
    for path, name in zip(a.model, a.name):
        be = Backend(path, a.threads, 64)
        meta = ("fase2-ood", name, "torch-fp32", be.kind, a.threads)
        T = fm.fit_temperature(be(fg.choice_inputs(cc), 8), [r["gold_idx"] for r in cc])
        pa, pb = fm.fit_platt(be(fg.noul_inputs(cn), 1)[:, 0], [r["label"] for r in cn])
        ch = items[items.task == "choice"]; no = items[items.task == "noul"]
        for lang in ("it", "en"):
            lg = be([fg.fmt(t, o) for t, o in zip(ch.text, ch[f"options_{lang}"])], 8)
            raw[f"{name}_choice_{lang}"] = lg
            for sc in sorted(ch.scenario.unique()):
                m = (ch.scenario == sc).values; y = ch.gold_idx.values[m]
                met = fm.choice_metrics(fm.softmax(lg[m], T), y)
                corr = (lg[m].argmax(1) == y).astype(float); ids = ch.id.values[m]
                met["accuracy_ci_lo"], met["accuracy_ci_hi"] = fm.bootstrap_ci(lambda ix: corr[ix].mean(), ids)
                rows += fm.rows(*meta, "choice", sc, f"options_{lang}", met, int(m.sum()))
            s = be([fg.fmt(t, [l]) for t, l in zip(no.text, no[f"label_{lang}"])], 1)[:, 0]
            raw[f"{name}_noul_{lang}"] = s
            p = fm.sigmoid(pa * s + pb)
            for sc in sorted(no.scenario.unique()):
                for sub in ("all",) + tuple(sorted(no[no.scenario == sc].neg_type.unique())) if sc == "domotica" else ("all",):
                    m = (no.scenario == sc).values
                    if sub != "all":
                        m = m & no.neg_type.isin(["pos", sub]).values
                    if sub == "pos":
                        continue
                    y = no.label.values[m]; sm = s[m]; ids = no.id.values[m]
                    met = fm.noul_metrics(p[m], y)
                    met["auroc_ci_lo"], met["auroc_ci_hi"] = fm.bootstrap_ci(lambda ix: roc_auc_score(y[ix], sm[ix]), ids)
                    rows += fm.rows(*meta, "noul", sc if sub == "all" else f"{sc}/{sub}", f"label_{lang}", met, int(m.sum()))
                if sc != "domotica":   # AUROC per singola domanda (proprietà)
                    for q in sorted(no[no.scenario == sc].q.unique()):
                        m = ((no.scenario == sc) & (no.q == q)).values
                        if len(set(no.label.values[m])) == 2:
                            rows += fm.rows(*meta, "noul", f"{sc}/q={q}", f"label_{lang}",
                                            {"auroc": roc_auc_score(no.label.values[m], s[m]),
                                             "pos_rate": float(no.label.values[m].mean())}, int(m.sum()))
        print(name, "fatto", flush=True)
    df = pd.DataFrame(rows, columns=fm.METRIC_COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    np.savez_compressed(os.path.splitext(a.out)[0] + "_scores.npz", **raw)
    k = df[df.metric.isin(["accuracy", "auroc", "ece"]) & ~df.split.str.contains("q=")]
    print(k.pivot_table(index=["task", "split", "lang"], columns=["model", "metric"], values="value").round(3).to_string())


if __name__ == "__main__":
    main()
