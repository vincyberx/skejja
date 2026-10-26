"""Qualità e tempi degli insegnanti nella prova di etichettatura (fase 2, dati di ampiezza).

Misure, per ogni insegnante:
- formato: quanta probabilità del primo token cade su una risposta valida (lettera o yes/no);
- metro (etichetta vera, split di training di SST-2, AG News, Emotion, SMS Spam; solo per misurare): accuratezza,
  ECE, e per lo spam AUROC e F1 con soglia 0,5; separato fra domanda in inglese e domanda nelle altre lingue;
- corpus (famiglia `lingua`, etichetta = lingua dichiarata dal corpus): accuratezza;
- stabilità: stessa richiesta con le opzioni in ordine inverso, e con la domanda nell'altra lingua;
- pool: distribuzione delle etichette e della confidenza per famiglia;
- accordo fra i due insegnanti sulle richieste del pool;
- tempi: richieste al secondo e stima per pool più grandi.

uso: python bench/teacher_analysis.py --requests bench/data/broad/teacher_trial_in.jsonl \
        --labels labels_qwen27b.jsonl labels_qwen9b.jsonl --tempi labels_qwen27b_tempi.json labels_qwen9b_tempi.json \
        --out results/fase2/teacher_trial.csv
"""
import argparse, collections, csv, json, os
import numpy as np


def ece(conf, ok, bins=15):
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)


def auroc(s, y):
    from scipy.stats import rankdata
    s, y = np.asarray(s, float), np.asarray(y, int)
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)) if n0 and n1 else float("nan")


def argmax(lab):
    return max(lab["p"], key=lab["p"].get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--tempi", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {}
    for l in open(a.requests):
        r = json.loads(l); R[r["rid"]] = r
    righe, riep = [], {}

    def add(teacher, gruppo, sotto, metrica, valore, n):
        righe.append(dict(insegnante=teacher, gruppo=gruppo, sottogruppo=sotto, metrica=metrica,
                          valore=None if valore is None or (isinstance(valore, float) and np.isnan(valore))
                          else round(float(valore), 4), n=n))

    tutte = {}
    for path in a.labels:
        L = {}
        for l in open(path):
            x = json.loads(l); L[x["rid"]] = x
        t = next(iter(L.values()))["modello"]
        tutte[t] = L
        rs = [(R[k], v) for k, v in L.items() if k in R]
        # formato
        for tipo in ("choice", "noul"):
            cop = [v["copertura"] for r, v in rs if r["tipo"] == tipo]
            add(t, "formato", tipo, "copertura_media", np.mean(cop), len(cop))
            add(t, "formato", tipo, "quota_copertura_sotto_0.9", np.mean(np.array(cop) < 0.9), len(cop))
        # metro
        for ds in ("sst2", "agnews", "emotion", "sms"):
            for chi, filtro in (("tutte", lambda r: True), ("domanda_en", lambda r: r["lang_domanda"] == "en"),
                                ("domanda_altre", lambda r: r["lang_domanda"] != "en")):
                m = [(r, v) for r, v in rs if r["gold_da"] == f"metro:{ds}" and filtro(r)]
                if not m:
                    continue
                if ds == "sms":
                    y = np.array([r["gold"] for r, _ in m]); p = np.array([v["p_yes"] for _, v in m])
                    pred = (p >= 0.5).astype(int)
                    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
                    add(t, "metro", f"{ds}/{chi}", "auroc", auroc(p, y), len(m))
                    add(t, "metro", f"{ds}/{chi}", "f1_soglia_0.5", 2 * tp / max(2 * tp + fp + fn, 1), len(m))
                    add(t, "metro", f"{ds}/{chi}", "accuratezza", (pred == y).mean(), len(m))
                    conf = np.where(pred == 1, p, 1 - p)
                    add(t, "metro", f"{ds}/{chi}", "ece", ece(conf, pred == y), len(m))
                else:
                    ok = np.array([argmax(v) == r["gold"] for r, v in m])
                    conf = np.array([max(v["p"].values()) for _, v in m])
                    add(t, "metro", f"{ds}/{chi}", "accuratezza", ok.mean(), len(m))
                    add(t, "metro", f"{ds}/{chi}", "ece", ece(conf, ok), len(m))
                    if ds == "sst2":
                        add(t, "metro", f"{ds}/{chi}", "quota_neutro", np.mean([argmax(v) == "neutro" for _, v in m]), len(m))
        # corpus: famiglia lingua
        for tipo in ("choice", "noul"):
            m = [(r, v) for r, v in rs if r["gold_da"] == "corpus" and r["tipo"] == tipo and r["uso"] == "addestramento"]
            if not m:
                continue
            if tipo == "choice":
                ok = np.array([argmax(v) == r["gold"] for r, v in m])
            else:
                ok = np.array([(v["p_yes"] >= 0.5) == bool(r["gold"]) for r, v in m])
            add(t, "corpus_lingua", tipo, "accuratezza", ok.mean(), len(m))
        # stabilità
        for uso in ("variante_ordine", "variante_lingua"):
            for tipo in ("choice", "noul"):
                m = [(r, v, L.get(r["di"])) for r, v in rs if r["uso"] == uso and r["tipo"] == tipo and r["di"] in L]
                if not m:
                    continue
                if tipo == "choice":
                    agree = np.mean([argmax(v) == argmax(o) for _, v, o in m])
                    tv = np.mean([0.5 * sum(abs(v["p"][k] - o["p"].get(k, 0)) for k in v["p"]) for _, v, o in m])
                    add(t, "stabilita", f"{uso}/{tipo}", "accordo_argmax", agree, len(m))
                    add(t, "stabilita", f"{uso}/{tipo}", "distanza_tv_media", tv, len(m))
                else:
                    agree = np.mean([(v["p_yes"] >= 0.5) == (o["p_yes"] >= 0.5) for _, v, o in m])
                    d = np.mean([abs(v["p_yes"] - o["p_yes"]) for _, v, o in m])
                    add(t, "stabilita", f"{uso}/{tipo}", "accordo_soglia_0.5", agree, len(m))
                    add(t, "stabilita", f"{uso}/{tipo}", "diff_p_media", d, len(m))
        # pool: etichette e confidenza per famiglia
        pool = [(r, v) for r, v in rs if r["uso"] == "addestramento" and r["gold_da"] == "insegnante"]
        dist = {}
        for fam in sorted({r["famiglia"] for r, _ in pool}):
            for tipo in ("choice", "noul"):
                m = [(r, v) for r, v in pool if r["famiglia"] == fam and r["tipo"] == tipo]
                if not m:
                    continue
                if tipo == "choice":
                    conf = np.array([max(v["p"].values()) for _, v in m])
                    add(t, "pool", f"{fam}/choice", "quota_conf_almeno_0.8", np.mean(conf >= 0.8), len(m))
                    dist[f"{fam}/choice"] = dict(collections.Counter(argmax(v) for _, v in m).most_common())
                else:
                    p = np.array([v["p_yes"] for _, v in m])
                    add(t, "pool", f"{fam}/noul", "quota_si", np.mean(p >= 0.5), len(m))
                    add(t, "pool", f"{fam}/noul", "quota_incerte_0.2_0.8", np.mean((p > 0.2) & (p < 0.8)), len(m))
            for lang in ("en", "it", "es", "de", "fr"):
                m = [(r, v) for r, v in pool if r["famiglia"] == fam and r["lang"] == lang and r["tipo"] == "choice"]
                if m:
                    add(t, "pool_lingua", f"{fam}/{lang}", "quota_conf_almeno_0.8",
                        np.mean([max(v["p"].values()) >= 0.8 for _, v in m]), len(m))
        riep[t] = dict(distribuzione_etichette_choice=dist)
    # accordo fra insegnanti
    nomi = sorted(tutte)
    if len(nomi) == 2:
        A, B = (tutte[n] for n in nomi)
        comuni = [k for k in A if k in B and R[k]["uso"] == "addestramento" and R[k]["gold_da"] == "insegnante"]
        for fam in sorted({R[k]["famiglia"] for k in comuni}):
            for tipo in ("choice", "noul"):
                ks = [k for k in comuni if R[k]["famiglia"] == fam and R[k]["tipo"] == tipo]
                if not ks:
                    continue
                if tipo == "choice":
                    v = np.mean([argmax(A[k]) == argmax(B[k]) for k in ks])
                else:
                    v = np.mean([(A[k]["p_yes"] >= 0.5) == (B[k]["p_yes"] >= 0.5) for k in ks])
                add(" vs ".join(nomi), "accordo_insegnanti", f"{fam}/{tipo}", "accordo", v, len(ks))
    # tempi
    for p in a.tempi:
        if not os.path.isfile(p):
            continue
        tm = json.load(open(p)); t = tm["modello"]
        add(t, "tempi", "generazione", "richieste_al_secondo", tm["richieste_al_secondo"], tm["richieste"])
        add(t, "tempi", "generazione", "token_prompt_al_secondo", tm["token_prompt_al_secondo"], tm["richieste"])
        add(t, "tempi", "caricamento", "secondi", tm["caricamento_s"], 1)
        for n in (100_000, 300_000, 1_000_000):
            add(t, "tempi", f"stima_{n}_richieste", "ore", n / tm["richieste_al_secondo"] / 3600, n)
        riep.setdefault(t, {})["tempi"] = {k: v for k, v in tm.items() if k != "blocchi"}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0])); w.writeheader(); w.writerows(righe)
    json.dump(riep, open(os.path.splitext(a.out)[0] + "_riepilogo.json", "w"), indent=1, ensure_ascii=False)
    for r in righe:
        if r["gruppo"] in ("metro", "corpus_lingua", "stabilita", "formato", "tempi", "accordo_insegnanti"):
            print(f"{r['insegnante'][:34]:<35}{r['gruppo']:<19}{r['sottogruppo']:<34}{r['metrica']:<26}{r['valore']}  (n={r['n']})")


if __name__ == "__main__":
    main()
