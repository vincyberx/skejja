"""Valutazione dei controlli sulle azioni di un agente sul test tldr (bench/data/agent/tldr_test.jsonl: programmi mai
visti in addestramento, etichette per regole). AUROC per proprietà e complessiva, con intervallo bootstrap per pagina
tldr (programma). Varianti di testo: solo comando (domanda in inglese o in italiano) e solo descrizione nella lingua della
traduzione (domanda nella stessa lingua; "sudo" escluso perché dipende dal comando).
ECE con la Platt globale del protocollo (calib_noul.jsonl): il caso "domanda senza esempi".

uso: python training/eval_agent.py --model releases/checkpoints/run_D/best --name edge32k-ft4 [--model ... --name ...] \
        --out results/fase2/metrics_agent_tldr.csv
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    D = a.data
    props = json.load(open(f"{D}/agent/props.json"))["props"]
    test = read(f"{D}/agent/tldr_test.jsonl")
    items = []
    for r in test:
        for k, v in sorted(r["labels"].items()):
            for ql in ("en", "it"):
                items.append(dict(variant=f"cmd|q_{ql}", page=r["page"], prop=k, label=v, text=r["cmd"], q=props[k][ql][0]))
            if k != "sudo":
                for l, d in r["desc"].items():
                    items.append(dict(variant=f"desc_{'en' if l == 'en' else 'native'}", page=r["page"], prop=k, label=v,
                                      text=d, q=props[k][l][0]))
    it = pd.DataFrame(items)
    cn = read(f"{D}/calib_noul.jsonl")
    rows = []
    for path, name in zip(a.model, a.name):
        be = Backend(path, a.threads, 64)
        pa, pb = fm.fit_platt(be(fg.noul_inputs(cn), 1)[:, 0], [r["label"] for r in cn])
        s = be([fg.fmt(t, [q]) for t, q in zip(it.text, it.q)], 1)[:, 0]
        it[name] = s
        meta = ("fase2-agent", name, "torch-fp32", be.kind, a.threads)
        for var in sorted(it.variant.unique()):
            d = it[it.variant == var]
            for k in ["all"] + sorted(d.prop.unique()):
                dd = d if k == "all" else d[d.prop == k]
                if dd.label.nunique() < 2:
                    continue
                y = dd.label.values; sv = dd[name].values; grp = dd.page.values
                m = fm.noul_metrics(fm.sigmoid(pa * sv + pb), y)
                m["auroc_ci_lo"], m["auroc_ci_hi"] = fm.bootstrap_ci(lambda ix: roc_auc_score(y[ix], sv[ix]), grp)
                m["pos_rate"] = float(y.mean())
                rows += fm.rows(*meta, "noul", f"tldr_test/{k}", var, m, len(dd))
        print(name, "fatto", flush=True)
    df = pd.DataFrame(rows, columns=fm.METRIC_COLUMNS)
    df.to_csv(a.out, index=False)
    # punteggi per esempio (dal 16 ottobre 2026): servono al test appaiato della famiglia B nei verdetti dei giri
    # (bench/verdetto_giro.py); l'ordine degli item è deterministico dato bench/data/agent
    np.savez_compressed(os.path.splitext(a.out)[0] + "_scores.npz",
                        variant=np.asarray(it.variant.tolist(), dtype="U"), page=np.asarray(it.page.tolist(), dtype="U"),
                        prop=np.asarray(it.prop.tolist(), dtype="U"), label=np.asarray(it.label.tolist(), dtype=int),
                        **{name: it[name].values.astype(np.float32) for name in a.name})
    k = df[df.metric.isin(["auroc", "ece"])]
    print(k.pivot_table(index=["split", "lang"], columns=["model", "metric"], values="value").round(3).to_string())


if __name__ == "__main__":
    main()
