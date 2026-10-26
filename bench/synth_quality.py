"""Rapporto di qualità delle etichette dell'insegnante sui testi di ampiezza.

Applica gli stessi filtri di training/train_skejja2.py (copertura minima, accordo con la classe voluta
quando il testo è stato generato per una classe) e conta, per famiglia, tipo e lingua:
  - richieste, senza etichetta, copertura bassa, disaccordo con la classe voluta, tenute;
  - accordo insegnante / classe voluta (solo dove la classe voluta esiste);
  - distribuzione delle etichette tenute (argmax per Choice, quota di sì per sì/no);
  - quota di etichette incerte (confidenza < 0,8 per Choice, 0,2 < p < 0,8 per sì/no).

Uso:
  python bench/synth_quality.py --requests bench/data/broad/requests_synth.jsonl.gz \
      --labels labels_synth_qwen9b.jsonl --out results/fase2/qualita_ampiezza.csv
"""
import argparse, collections, gzip, json

import numpy as np


def apri(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--teacher", default="")
    ap.add_argument("--min_cop", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L = {}
    for p in a.labels:
        for l in apri(p):
            x = json.loads(l)
            if a.teacher and x.get("modello") != a.teacher:
                continue
            L[x["rid"]] = x

    C = collections.defaultdict(collections.Counter)      # (fam, tipo, lingua) -> contatori
    acc = collections.defaultdict(list)                   # accordo con la classe voluta
    incerte = collections.defaultdict(list)
    dist = collections.defaultdict(collections.Counter)   # etichette tenute
    tot = collections.Counter()
    for p in a.requests:
        for l in apri(p):
            r = json.loads(l)
            if r.get("uso") != "addestramento" or str(r.get("gold_da", "")).startswith("metro"):
                continue
            chiavi = [(r["famiglia"], r["tipo"], r["lang"]), (r["famiglia"], r["tipo"], "tutte"),
                      ("tutte", r["tipo"], "tutte")]
            org = r.get("origine", "naturale")
            for k in chiavi:
                C[k]["richieste"] += 1
            tot[f"richieste/{org}"] += 1
            lab = L.get(r["rid"])
            if lab is None:
                for k in chiavi:
                    C[k]["senza_etichetta"] += 1
                continue
            if lab["copertura"] < a.min_cop:
                for k in chiavi:
                    C[k]["copertura_bassa"] += 1
                continue
            if r["tipo"] == "choice":
                pv = np.array([lab["p"].get(o, 0.0) for o in r["opzioni"]]); pv = pv / max(pv.sum(), 1e-9)
                pred = r["opzioni"][int(np.argmax(pv))]; conf = float(pv.max())
                ok = None if r.get("atteso") is None else (pred == r["atteso"])
                inc = conf < 0.8
            else:
                py = float(lab["p_yes"]); pred = "si" if py >= 0.5 else "no"
                ok = None if r.get("atteso") is None else (int(py >= 0.5) == int(r["atteso"]))
                inc = 0.2 < py < 0.8
            for k in chiavi:
                if ok is not None:
                    acc[k].append(ok)
                if ok is False:
                    C[k]["disaccordo_con_classe_voluta"] += 1
                    continue
                C[k]["tenute"] += 1
                incerte[k].append(inc)
                dist[k][pred] += 1
            tot[f"tenute/{org}"] += int(ok is not False)

    righe = []
    for k in sorted(C):
        fam, tipo, lingua = k
        c = C[k]
        d = dist[k]; n_t = sum(d.values())
        if tipo == "choice":
            top = d.most_common(1)[0] if d else ("", 0)
            distr = dict(classe_piu_frequente=top[0], quota_classe_piu_frequente=round(top[1] / max(n_t, 1), 4),
                         classi_distinte=len(d))
        else:
            distr = dict(classe_piu_frequente="si", quota_classe_piu_frequente=round(d["si"] / max(n_t, 1), 4),
                         classi_distinte=len(d))
        righe.append(dict(famiglia=fam, tipo=tipo, lingua=lingua, richieste=c["richieste"],
                          senza_etichetta=c["senza_etichetta"], copertura_bassa=c["copertura_bassa"],
                          disaccordo_con_classe_voluta=c["disaccordo_con_classe_voluta"], tenute=c["tenute"],
                          n_con_classe_voluta=len(acc[k]),
                          accordo_con_classe_voluta=round(float(np.mean(acc[k])), 4) if acc[k] else None,
                          quota_incerte=round(float(np.mean(incerte[k])), 4) if incerte[k] else None, **distr))
    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0])); w.writeheader(); w.writerows(righe)
    riep = dict(totali=dict(tot), etichette=len(L), min_cop=a.min_cop,
                per_famiglia={f"{r['famiglia']}/{r['tipo']}": dict(tenute=r["tenute"], accordo=r["accordo_con_classe_voluta"],
                                                                   classe_piu_frequente=r["classe_piu_frequente"],
                                                                   quota=r["quota_classe_piu_frequente"])
                              for r in righe if r["lingua"] == "tutte"})
    json.dump(riep, open(a.out.replace(".csv", "_riepilogo.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(totali=dict(tot), righe=len(righe))))


if __name__ == "__main__":
    main()
