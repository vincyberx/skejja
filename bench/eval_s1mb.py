"""Aggrega i risultati S1MB di Skejja e li confronta con i modelli della classifica pubblica.

Legge i file di risultato prodotti dall'harness di S1MB (un JSON per prova, vedi bench/s1mb_adapter/) e calcola
le medie per famiglia di domande con la stessa regola della classifica: media del punteggio corretto per la
linea di base (`baseline_adjusted_score`) sulle prove ammesse (`scoring.eligible`). La «Task Avg» della
classifica è la media NON pesata delle tre medie di famiglia: si riporta solo se le tre famiglie sono misurate e
nessuna decisione è fallita (fino a ft6 Skejja non aveva la primitiva `Score` e le 21 prove risultavano fallite).

Riporta anche, separatamente, le prove per cui NON dichiariamo esposizione dei dati di addestramento, così il
confronto con Bekko (che dichiara esposizione su 98 prove su 137) è leggibile.

uso: python bench/eval_s1mb.py --results <dir con i JSON> --name edge32k-ft5 \
        --stats <file stats*.json ...> --out results/fase2/metrics_s1mb.csv
"""
import argparse, csv, glob, json, os, statistics
from collections import defaultdict

RIFERIMENTI = os.path.join("results", "fase2", "s1mb_riferimenti.csv")


def leggi(dirs):
    out = {}
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = json.load(open(path))
            if "benchmark" not in r or "metrics" not in r:
                continue   # segnaposto o file non di risultato
            out[r["benchmark"]["id"]] = r   # un file per prova; l'ultimo vince
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--stats", nargs="*", default=[])
    ap.add_argument("--riferimenti", default=RIFERIMENTI)
    ap.add_argument("--out", default=os.path.join("results", "fase2", "metrics_s1mb.csv"))
    a = ap.parse_args()
    res = leggi(a.results)
    assert res, "nessun risultato trovato"
    esposizione = {e["benchmark_id"] for r in res.values()
                   for e in (r["model"].get("training_exposure") or [])}
    righe, per_famiglia, senza_esposizione = [], defaultdict(list), defaultdict(list)
    for bid, r in sorted(res.items()):
        b, m = r["benchmark"], r["metrics"]
        punteggio = m.get("baseline_adjusted_score")
        ammessa = bool(r.get("scoring", {}).get("eligible", punteggio is not None))
        righe.append(dict(
            modello=a.name, prova=bid, famiglia=b["task"], casi=r["counts"]["cases"],
            decisioni_attese=r["counts"]["expected"], riuscite=r["counts"]["succeeded"],
            fallite=r["counts"]["failed"], stato=r["status"],
            metrica_primaria=b["primary_metric"], valore_metrica=m.get(b["primary_metric"]),
            punteggio_corretto=punteggio, ammessa=int(ammessa),
            nostra_esposizione=int(bid in esposizione), secondi=round(r["elapsed_seconds"], 1)))
        if ammessa and punteggio is not None:
            per_famiglia[b["task"]].append(punteggio)
            if bid not in esposizione:
                senza_esposizione[b["task"]].append(punteggio)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cols = list(righe[0])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(righe)

    media = lambda v: round(100 * statistics.mean(v), 2) if v else None
    stats = defaultdict(int)
    for s in a.stats:
        if os.path.isfile(s):
            for k, v in json.load(open(s)).items():
                stats[k] = max(stats[k], v) if k.endswith("_max") else stats[k] + v
    conf = {}
    if os.path.isfile(a.riferimenti):
        for r in csv.DictReader(open(a.riferimenti)):
            conf[r["modello"]] = {k: r[k] for k in ("noul", "choice", "score", "task_avg",
                                                    "prove_con_esposizione_dichiarata")}
    riepilogo = dict(
        modello=a.name, prove_misurate=len(righe),
        prove_per_famiglia={t: sum(1 for r in righe if r["famiglia"] == t) for t in ("noul", "choice", "score")},
        decisioni_riuscite=sum(r["riuscite"] for r in righe),
        decisioni_fallite=sum(r["fallite"] for r in righe),
        score=dict(
            prove=sum(1 for r in righe if r["famiglia"] == "score"),
            decisioni=sum(r["decisioni_attese"] for r in righe if r["famiglia"] == "score"),
            fallite=sum(r["fallite"] for r in righe if r["famiglia"] == "score"),
            nota="con fallite > 0 la primitiva Score manca o non regge: quelle decisioni sono fallite, non sbagliate, "
                 "e la Task Avg non è confrontabile (fino al 15 ottobre questo campo si chiamava score_non_supportata "
                 "e diceva «Skejja non ha la primitiva Score» anche quando c'era)"),
        medie_per_famiglia={t: media(v) for t, v in sorted(per_famiglia.items())},
        task_avg=(round(statistics.mean(media(per_famiglia[t]) for t in ("noul", "choice", "score")), 2)
                  if all(per_famiglia.get(t) for t in ("noul", "choice", "score"))
                  and sum(r["fallite"] for r in righe) == 0 else None),
        medie_senza_nostra_esposizione={t: media(v) for t, v in sorted(senza_esposizione.items())},
        prove_con_nostra_esposizione=sorted(b for b in esposizione if b in res),
        lunghezze_ingresso=dict(stats) or None,
        confronto_classifica=conf)
    json.dump(riepilogo, open(os.path.splitext(a.out)[0] + "_riepilogo.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in riepilogo.items() if k != "confronto_classifica"}, indent=1,
                     ensure_ascii=False))
    if conf:
        print(f"\n{'modello':<40}{'noul':>8}{'choice':>8}{'score':>8}{'esposiz.':>10}")
        nostre = riepilogo["medie_per_famiglia"]
        print(f"{'Skejja ' + a.name:<40}{str(nostre.get('noul')):>8}{str(nostre.get('choice')):>8}"
              f"{'n.d.':>8}{len(riepilogo['prove_con_nostra_esposizione']):>10}")
        for mod, v in sorted(conf.items(), key=lambda x: -float(x[1]["choice"])):
            print(f"{mod:<40}{v['noul']:>8}{v['choice']:>8}{v['score']:>8}"
                  f"{v['prove_con_esposizione_dichiarata']:>10}")


if __name__ == "__main__":
    main()
