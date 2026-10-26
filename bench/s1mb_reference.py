"""Numeri di riferimento della classifica S1MB (solo misure pubblicate da altri, nessuna misura nostra).

Legge `viewer-summary.json` del dataset pubblico `hotchpotch/s1mb-result` a una revisione fissata e ricava, per i
modelli che ci interessano, la media del punteggio corretto per la linea di base (`baseline_adjusted_score`) per
famiglia di domande. Verificato: la «Task Avg» della classifica è la media NON pesata delle tre medie di famiglia
(noul, choice, score), non la media sulle 137 prove; per bekko-68m (42,91 + 51,62 + 26,85) / 3 = 40,46.

Riporta anche `training_exposure`, cioè le prove per cui un modello dichiara di avere la stessa famiglia di
sorgenti nei dati di addestramento: senza questa colonna i confronti su S1MB non sono leggibili.

uso: python bench/s1mb_reference.py --out results/fase2/s1mb_riferimenti.csv
"""
import argparse, csv, json, os, statistics, urllib.request
from collections import defaultdict

REPO = "hotchpotch/s1mb-result"
CATEGORIA = "english-v1"
# Modelli di confronto: la nostra fascia di dimensione (Bekko), i System One aperti che già citiamo e Jev.
MODELLI = ["hotchpotch__bekko-system-one-v0-17m", "hotchpotch__bekko-system-one-v0-68m",
           "hotchpotch__bekko-system-one-v0-400m", "convaiinnovations__laya-multilingual",
           "convaiinnovations__laya", "convaiinnovations__laya-typed-decisions",
           "fastino__GLiNER2.5-Decide", "jaredpalmer__kev-0.8b", "jaredpalmer__kev-4b",
           "MoritzLaurer__ModernBERT-large-zeroshot-v2.0", "typesafe__jev_1_13"]
# Le nostre sorgenti di addestramento (fase 1 e 2), per dichiarare la nostra esposizione alle prove di S1MB.
NOSTRE_SORGENTI = ["massive", "clinc", "banking77", "minds14", "tldr"]


def scarica(path, revisione, dest=None):
    """La cache è per revisione: un file già scaricato non viene mai riusato per una revisione diversa."""
    dest = dest or f"/tmp/s1mb_{revisione[:12]}_{path.replace('/', '_')}"
    if not os.path.exists(dest):
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/{revisione}/{path}", dest)
    return dest


def revisione_corrente():
    d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=60))
    return d["sha"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="file già scaricato (si usa solo se corrisponde alla revisione)")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", default="results/fase2/s1mb_riferimenti.csv")
    a = ap.parse_args()
    rev = revisione_corrente() if a.revision == "main" else a.revision
    d = json.load(open(scarica("viewer-summary.json", rev, a.cache)))
    # `source.revision` dentro il file è la revisione dei RISULTATI aggregati, che può essere più vecchia della
    # revisione del repository da cui scarichiamo il file: le registriamo entrambe, mai una sola.
    rev_risultati = d.get("source", {}).get("revision")
    p = d["payload"]
    B, M = p["benchmarks"], p["models"]
    cat = {c["id"]: set(c["benchmarks"]) for c in p["categories"]}[CATEGORIA]
    exposure = defaultdict(set)
    for m in M:
        for e in m["model"].get("training_exposure", []) or []:
            exposure[m["run_id"]].add(e["benchmark_id"])
    val = defaultdict(lambda: defaultdict(list))
    for r in p["results"]:
        rid = M[r["model"]]["run_id"]
        b = B[r["benchmark"]]
        if b["id"] not in cat or not r.get("scoring", {}).get("eligible"):
            continue
        s = r["metrics"].get("baseline_adjusted_score")
        if s is not None:
            val[rid][b["task"]].append(s)
    info = {m["run_id"]: m["model"] for m in M}
    righe = []
    for rid in MODELLI:
        if rid not in val:
            print("manca:", rid)
            continue
        mm = info[rid]
        fam = {t: 100 * statistics.mean(val[rid][t]) for t in ("noul", "choice", "score") if val[rid].get(t)}
        righe.append(dict(
            run_id=rid, modello=mm.get("id"), parametri_totali=mm.get("total_params"),
            parametri_non_lookup=mm.get("active_params"), metodo_conteggio=mm.get("parameter_count_method"),
            prove=sum(len(v) for v in val[rid].values()),
            task_avg=round(statistics.mean(fam.values()), 2) if len(fam) == 3 else "",
            noul=round(fam.get("noul", float("nan")), 2), choice=round(fam.get("choice", float("nan")), 2),
            score=round(fam.get("score", float("nan")), 2),
            prove_con_esposizione_dichiarata=len(exposure.get(rid, set()) & cat)))
    cols = list(righe[0])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(righe, key=lambda r: -(r["task_avg"] or 0)))
    nostra = sorted(b["id"] for b in B if b["id"] in cat
                    and any(s in b["dataset"].rsplit("/", 1)[-1] for s in NOSTRE_SORGENTI))
    riepilogo = dict(
        dataset=REPO, revisione_file_scaricato=rev, revisione_dei_risultati_aggregati=rev_risultati,
        categoria=CATEGORIA, generato=d.get("generatedAt"),
        prove=len(cat), modelli_nella_classifica=len(M),
        prove_per_famiglia={t: sum(1 for b in B if b["id"] in cat and b["task"] == t)
                            for t in sorted({b["task"] for b in B})},
        metrica_primaria={t: sorted({b["primary_metric"] for b in B if b["id"] in cat and b["task"] == t})
                          for t in sorted({b["task"] for b in B})},
        task_avg="media non pesata delle tre medie di famiglia (verificato su bekko-68m)",
        prove_con_sorgenti_nostre=nostra,
        esposizione_bekko_68m=len(exposure["hotchpotch__bekko-system-one-v0-68m"] & cat))
    json.dump(riepilogo, open(os.path.splitext(a.out)[0] + "_riepilogo.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(riepilogo, indent=1, ensure_ascii=False))
    for r in sorted(righe, key=lambda r: -(r["task_avg"] or 0)):
        print(f"{r['modello']:<46}{str(r['task_avg']):>7}{r['noul']:>8.2f}{r['choice']:>8.2f}{r['score']:>8.2f}"
              f"   esposizione {r['prove_con_esposizione_dichiarata']:>3}/{len(cat)}")


if __name__ == "__main__":
    main()
