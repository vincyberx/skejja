"""Qualità dei dati del giro G: etichette dell'insegnante, etichette dei corpora, lunghezze e opzioni.

Applica gli stessi filtri dell'addestramento (copertura ≥ `--min_cop`, si scarta la richiesta se l'insegnante
non conferma la classe voluta) e scrive una tabella per famiglia più un riepilogo. Controlla anche che nessun
testo delle richieste coincida con il banco di prova o con gli stati di S1MB.

uso: python bench/qualita_giroG.py --richieste bench/data/broad/richieste_giroG.jsonl \\
         --etichette bench/data/broad/labels_giroG_qwen9b.jsonl --out results/fase2/qualita_giroG.csv
"""

import argparse
import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_broad import chiave, testi_del_progetto  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--richieste", required=True)
    ap.add_argument("--etichette", default="")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default="results/fase2/qualita_giroG.csv")
    ap.add_argument("--min_cop", type=float, default=0.9)
    ap.add_argument("--filtra", default="", help="scrive qui le richieste che NON toccano il banco di prova")
    a = ap.parse_args()

    lab = {}
    if a.etichette:
        for l in open(a.etichette):
            x = json.loads(l)
            lab.setdefault(x["rid"], x)

    # Insieme proibito: test, sviluppo, calibrazione, test delle azioni, metro dell'insegnante e stati di S1MB.
    # NON comprende `train_utterances.jsonl`: le frasi di addestramento delle intenzioni sono nostre e le Choice
    # con molte opzioni nascono proprio da lì (toglierle sarebbe un falso positivo).
    visti = testi_del_progetto(a.data)
    addestramento = set()
    for l in open(os.path.join(a.data, "train_utterances.jsonl")):
        r = json.loads(l)
        if r.get("text"):
            addestramento.add(chiave(r["text"]))
    visti -= addestramento
    s1mb = json.load(open(os.path.join(a.data, "broad", "s1mb_testi_hash.json")))["hash"]
    visti |= set(s1mb)

    fam = collections.defaultdict(lambda: collections.Counter())
    livelli = collections.defaultdict(collections.Counter)
    token, sovrapposti, incerte = [], 0, collections.Counter()
    tocca_per_famiglia = collections.Counter()
    pulite = open(a.filtra, "w") if a.filtra else None
    for l in open(a.richieste):
        r = json.loads(l)
        f = fam[(r["famiglia"], r["tipo"])]
        f["richieste"] += 1
        f["lingue_" + r["lang"]] += 1
        token.append(len(r["text"]) // 4)
        stato = r.get("stato") or {}
        tocca = any(isinstance(v, str) and len(v) >= 80 and chiave(v) in visti
                    for v in (stato.values() if isinstance(stato, dict) else [stato]))
        if tocca:
            sovrapposti += 1
            tocca_per_famiglia[r["famiglia"]] += 1
            f["tocca_il_banco"] += 1
            continue          # richiesta esclusa: la regola di separazione viene prima di tutto
        if pulite:
            pulite.write(l)
        if r["gold_da"] == "corpus":
            f["etichetta_corpus"] += 1
            f["tenute"] += 1
            if r["tipo"] == "score":
                livelli[r["famiglia"]][str(r["gold"])] += 1
            continue
        x = lab.get(r["rid"])
        if x is None:
            f["senza_etichetta"] += 1
            continue
        if x["copertura"] < a.min_cop:
            f["copertura_bassa"] += 1
            continue
        if r["tipo"] in ("choice", "score"):
            p = x["p"]
            top = max(p, key=p.get)
            if r.get("atteso") is not None and top != r["atteso"]:
                f["disaccordo_con_classe_voluta"] += 1
                continue
            if max(p.values()) < 0.8:
                incerte[r["famiglia"]] += 1
            if r["tipo"] == "score":
                livelli[r["famiglia"]][top] += 1
        else:
            if r.get("atteso") is not None and int(x["p_yes"] >= 0.5) != int(r["atteso"]):
                f["disaccordo_con_classe_voluta"] += 1
                continue
            if 0.2 < x["p_yes"] < 0.8:
                incerte[r["famiglia"]] += 1
            f["si"] += int(x["p_yes"] >= 0.5)
        f["etichetta_insegnante"] += 1
        f["tenute"] += 1

    righe = []
    for (nome, tipo), c in sorted(fam.items()):
        righe.append(dict(famiglia=nome, tipo=tipo, richieste=c["richieste"], tenute=c["tenute"],
                          etichetta_corpus=c["etichetta_corpus"], etichetta_insegnante=c["etichetta_insegnante"],
                          senza_etichetta=c["senza_etichetta"], copertura_bassa=c["copertura_bassa"],
                          disaccordo_con_classe_voluta=c["disaccordo_con_classe_voluta"],
                          quota_si=round(c["si"] / c["etichetta_insegnante"], 3) if c["etichetta_insegnante"] and tipo == "noul" else "",
                          incerte=incerte.get(nome, 0), tocca_il_banco=c["tocca_il_banco"]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0]))
        w.writeheader()
        w.writerows(righe)
    token.sort()
    riepilogo = dict(
        richieste=sum(r["richieste"] for r in righe), tenute=sum(r["tenute"] for r in righe),
        con_etichetta_del_corpus=sum(r["etichetta_corpus"] for r in righe),
        con_etichetta_dell_insegnante=sum(r["etichetta_insegnante"] for r in righe),
        scartate=dict(senza_etichetta=sum(r["senza_etichetta"] for r in righe),
                      copertura_bassa=sum(r["copertura_bassa"] for r in righe),
                      disaccordo_con_classe_voluta=sum(r["disaccordo_con_classe_voluta"] for r in righe)),
        token_stimati=dict(mediana=token[len(token) // 2], p90=token[int(len(token) * 0.9)], massimo=token[-1]),
        livelli_score={k: dict(v) for k, v in livelli.items()},
        testi_che_toccano_il_banco=sovrapposti,
        testi_che_toccano_il_banco_per_famiglia=dict(tocca_per_famiglia),
        file_pulito=a.filtra or None,
        nota="Il controllo di sovrapposizione confronta i pezzi di stato lunghi almeno 80 caratteri con i testi "
             "del progetto e con le impronte degli stati di S1MB.")
    if pulite:
        pulite.close()
    json.dump(riepilogo, open(os.path.splitext(a.out)[0] + "_riepilogo.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(riepilogo, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
