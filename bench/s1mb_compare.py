"""Confronto prova per prova su S1MB: Skejja contro i modelli della classifica, con metriche secondarie.

Il punteggio della classifica (`baseline_adjusted_score`) è tagliato a 0 sotto la linea di base: una media vicina
a zero non dice se il modello ordina bene ma ha probabilità mal calibrate, o se non capisce il compito. Per questo
riportiamo anche, per le Noul, l'AUROC calcolata dalle predizioni salvate e l'accuratezza bilanciata con soglia
0,5; per le Choice l'accuratezza (argmax) contro la linea di base uniforme. Nessuna di queste misure cambia la
configurazione di Skejja: sono diagnosi a configurazione fissata (politica sul test di S1MB).

uso: python bench/s1mb_compare.py --results <dir JSON ft5> --name edge32k-ft5 [--results <dir> --name ...] \
        --dataset <cartella data/ di S1MB con i target> --out results/fase2/s1mb_confronto_per_prova.csv
"""
import argparse, csv, glob, json, os, sys
import numpy as np

CONFRONTO = ["hotchpotch__bekko-system-one-v0-68m", "hotchpotch__bekko-system-one-v0-17m",
             "convaiinnovations__laya-multilingual", "fastino__GLiNER2.5-Decide", "typesafe__jev_1_13"]


def auroc(s, y):
    from scipy.stats import rankdata
    s, y = np.asarray(s, float), np.asarray(y, int)
    n1 = y.sum(); n0 = len(y) - n1
    if not n0 or not n1:
        return None
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def targets(data_dir, bid):
    """Target per (case_id, question_id) dalla cartella dati di S1MB (solo per la diagnosi a posteriori)."""
    sys.path.insert(0, os.path.join(data_dir, "..", "src"))
    from pathlib import Path
    from s1mb.data import Benchmark, load_cases, read_json
    b = Benchmark.model_validate(read_json(Path(data_dir) / "benchmarks" / f"{bid}.json"))
    out = {}
    for c in load_cases(Path(data_dir), b):
        for q in c.questions:
            out[(c.case_id, q.id)] = c.targets[q.id].probabilities
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True)
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--dataset", required=True, help="cartella evaluator/data di S1MB (benchmarks/, datasets/)")
    ap.add_argument("--viewer", default="/tmp/s1mb_viewer.json")
    ap.add_argument("--out", default="results/fase2/s1mb_confronto_per_prova.csv")
    a = ap.parse_args()
    V = json.load(open(a.viewer))["payload"]
    B, M = V["benchmarks"], V["models"]
    ref = {}
    for r in V["results"]:
        rid = M[r["model"]]["run_id"]
        if rid in CONFRONTO and r.get("scoring", {}).get("eligible"):
            ref[(rid, B[r["benchmark"]]["id"])] = r["metrics"].get("baseline_adjusted_score")
    nostri = {}
    for nome, d in zip(a.name, a.results):
        for f in glob.glob(os.path.join(d, "*.json")):
            x = json.load(open(f))
            if "benchmark" in x:
                nostri[(nome, x["benchmark"]["id"])] = x
    prove = sorted({b for _, b in nostri})
    righe = []
    for bid in prove:
        task = nostri[(a.name[0], bid)]["benchmark"]["task"]
        riga = dict(prova=bid, famiglia=task)
        tg = targets(a.dataset, bid) if task in ("noul", "choice") else {}
        for nome in a.name:
            x = nostri[(nome, bid)]
            m = x["metrics"]
            riga[f"{nome}_punteggio"] = None if m.get("baseline_adjusted_score") is None else round(100 * m["baseline_adjusted_score"], 2)
            riga[f"{nome}_metrica"] = m.get(x["benchmark"]["primary_metric"])
            preds = {(p["case_id"], p["question_id"]): p["probabilities"] for p in x["predictions"] if p.get("probabilities")}
            if task == "noul" and preds:
                ks = [k for k in preds if k in tg]
                y = np.array([tg[k]["true"] >= 0.5 for k in ks], int)
                s = np.array([preds[k]["true"] for k in ks])
                pred = s >= 0.5
                tpr = (pred & (y == 1)).sum() / max(1, (y == 1).sum()); tnr = (~pred & (y == 0)).sum() / max(1, (y == 0).sum())
                riga[f"{nome}_auroc"] = None if auroc(s, y) is None else round(auroc(s, y), 4)
                riga[f"{nome}_acc_bilanciata"] = round(float((tpr + tnr) / 2), 4)
                riga[f"{nome}_p_true_media"] = round(float(s.mean()), 4)
            if task == "choice" and preds:
                ks = [k for k in preds if k in tg]
                ok = [max(preds[k], key=preds[k].get) == max(tg[k], key=tg[k].get) for k in ks]
                riga[f"{nome}_accuratezza"] = round(float(np.mean(ok)), 4)
                riga[f"{nome}_base_uniforme"] = m.get("uniform_accuracy_baseline")
        for rid in CONFRONTO:
            v = ref.get((rid, bid))
            riga[rid.split("__")[1] + "_punteggio"] = None if v is None else round(100 * v, 2)
        righe.append(riga)
    cols = sorted({k for r in righe for k in r}, key=lambda k: (k not in ("prova", "famiglia"), k))
    cols = ["prova", "famiglia"] + [c for c in cols if c not in ("prova", "famiglia")]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(righe)
    # riepilogo diagnostico
    riep = {}
    for nome in a.name:
        nl = [r for r in righe if r["famiglia"] == "noul" and r.get(f"{nome}_auroc") is not None]
        ch = [r for r in righe if r["famiglia"] == "choice" and r.get(f"{nome}_accuratezza") is not None]
        bekko = "bekko-system-one-v0-68m_punteggio"
        riep[nome] = dict(
            noul_prove=len(nl),
            noul_auroc_mediana=float(np.median([r[f"{nome}_auroc"] for r in nl])),
            noul_prove_auroc_almeno_0_6=sum(r[f"{nome}_auroc"] >= 0.6 for r in nl),
            noul_prove_auroc_sotto_0_5=sum(r[f"{nome}_auroc"] < 0.5 for r in nl),
            noul_acc_bilanciata_mediana=float(np.median([r[f"{nome}_acc_bilanciata"] for r in nl])),
            noul_p_true_media_mediana=float(np.median([r[f"{nome}_p_true_media"] for r in nl])),
            noul_prove_punteggio_sopra_0=sum((r[f"{nome}_punteggio"] or 0) > 0 for r in nl),
            choice_prove=len(ch),
            choice_accuratezza_mediana=float(np.median([r[f"{nome}_accuratezza"] for r in ch])),
            choice_prove_sopra_base_uniforme=sum(r[f"{nome}_accuratezza"] > (r[f"{nome}_base_uniforme"] or 0) + 0.05 for r in ch),
            choice_prove_punteggio_sopra_0=sum((r[f"{nome}_punteggio"] or 0) > 0 for r in ch),
            prove_sopra_bekko_68m=sorted(r["prova"] for r in righe if r.get(f"{nome}_punteggio") is not None
                                          and r.get(bekko) is not None and r[f"{nome}_punteggio"] > r[bekko]),
        )
    json.dump(riep, open(os.path.splitext(a.out)[0] + "_riepilogo.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(riep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
