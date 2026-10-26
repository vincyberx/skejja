"""Verdetto di un giro sulle misure di guardia: test appaiati, Holm per famiglia, tolleranze del file obiettivi.

Legge l'insieme di guardia da results/fase2/giro_<X>_obiettivi.json (misure, verso, famiglia) e i punteggi per
esempio dei due modelli, poi per ogni misura calcola la differenza candidato − riferimento con un bootstrap
appaiato (10.000 ricampionamenti per default, come dichiara la regola), il p bilaterale dal bootstrap, la
correzione di Holm dentro la famiglia e il giudizio secondo la tolleranza della famiglia.

File dei punteggi (quelli che gli script di valutazione salvano accanto al CSV):
  A  training/eval_noul.py [--native]   metrics_<nome>_scores.npz          choice_test, choice_calib, test, ext, calib
                                                                          (+ choice_test_native se --native)
  B  training/eval_agent.py             metrics_agent_tldr_<x>_scores.npz  <nome>, variant, page, label
  C  training/eval_ood.py               metrics_ood_v0_<x>_scores.npz      <nome>_choice_it|en, <nome>_noul_it|en
  D  bench/eval_jev_suite.py            metrics_jev_suite_<x>_scores.npz   <nome>|<dataset>, y|<dataset>

Una misura senza punteggi per esempio risulta «non misurata» e la sua famiglia «incompleta»: non si ripiega sulle
stime puntuali.

uso: python bench/verdetto_giro.py --obiettivi results/fase2/giro_G_obiettivi.json --rif edge32k-ft6 --cand edge32k-ft7 \\
        --a_rif results/fase2/metrics_edge32k-ft6_scores.npz --a_cand results/fase2/metrics_edge32k-ft7_scores.npz \\
        --c_rif results/fase2/metrics_ood_v0_ft6_scores.npz --c_cand results/fase2/metrics_ood_v0_ft7_scores.npz \\
        --d_rif results/fase2/metrics_jev_suite_ft6_scores.npz --d_cand results/fase2/metrics_jev_suite_ft7_scores.npz \\
        --out results/fase2/verdetto_giro_G.csv
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skejja_measure as fm  # noqa: E402

JEV = {"agnews (accuracy)": "agnews", "rotten (accuracy)": "rotten", "sst2 (accuracy)": "sst2",
       "emotion (accuracy)": "emotion", "finphrase (accuracy)": "finphrase", "langid (accuracy)": "langid",
       "banking77 (accuracy)": "banking77", "sib200_eng (accuracy)": "sib200_eng", "sib200_ita (accuracy)": "sib200_ita",
       "sib200_spa (accuracy)": "sib200_spa", "sib200_deu (accuracy)": "sib200_deu", "sib200_fra (accuracy)": "sib200_fra",
       "sms (F1_pos)": "sms"}
IN_DOMINIO = {"banking77 (accuracy)"}   # «Banking77 è in dominio e va letta a parte» (regola del giro G)


def read(p):
    return [json.loads(l) for l in open(p)]


def gruppi_idx(groups):
    ug, inv = np.unique(groups, return_inverse=True)
    return ug, [np.where(inv == k)[0] for k in range(len(ug))]


def boot_appaiato(stat, groups, B, seed=0):
    """stat(ix) -> differenza orientata (positivo = meglio il candidato). Ricampiona i gruppi."""
    rng = np.random.default_rng(seed)
    ug, idx = gruppi_idx(np.asarray(groups))
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(ug), len(ug))
        vals[b] = stat(np.concatenate([idx[k] for k in pick]))
    return vals


def f1_pos(p, y):
    pred = p >= 0.5; tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn)


def misure_A(a, data):
    """Restituisce {nome misura: (fn(npz, ix) -> valore, gruppi, n)} per la famiglia A."""
    ch = read(f"{data}/test_choice.jsonl"); cc = read(f"{data}/calib_choice.jsonl")
    y = np.array([r["gold_idx"] for r in ch]); g = np.array([r["id"] for r in ch])
    lang = np.array([r["lang"] for r in ch]); uns = np.array([fm.is_unseen(r["gold"]) for r in ch])
    yc = np.array([r["gold_idx"] for r in cc])
    te = read(f"{data}/test_noul.jsonl"); ex = read(f"{data}/test_noul_unseen_ext.jsonl"); cn = read(f"{data}/calib_noul.jsonl")
    yt = np.array([r["label"] for r in te]); gt = np.array([str(r["id"]) for r in te])
    ye = np.array([r["label"] for r in ex]); ge = np.array([str(r["id"]) for r in ex]); yn = np.array([r["label"] for r in cn])
    cache = {}

    def T(z):     # temperatura sul set di calibrazione, una volta per modello (non ricampionata)
        k = ("T", id(z))
        if k not in cache:
            cache[k] = fm.fit_temperature(z["choice_calib"], yc)
        return cache[k]

    def platt(z):
        k = ("P", id(z))
        if k not in cache:
            cache[k] = fm.fit_platt(z["calib"], yn)
        return cache[k]

    def gap(z, ix, key="choice_test"):
        ok = z[key][ix].argmax(1) == y[ix]; l = lang[ix]
        acc = {L: ok[l == L].mean() for L in fm.LANGS}
        return acc["en"] - min(v for k_, v in acc.items() if k_ != "en")

    def ece_choice(z, ix):
        p = fm.softmax(z["choice_test"][ix], T(z)); return fm.ece(p.max(1), p.argmax(1) == y[ix])

    def ece_noul(z, ix):
        pa, pb = platt(z); p = fm.sigmoid(pa * z["test"][ix] + pb); pred = (p >= 0.5).astype(int)
        return fm.ece(np.where(pred == 1, p, 1 - p), pred == yt[ix])

    uix = np.where(uns)[0]
    return {
        "Choice, tutte": (lambda z, ix: (z["choice_test"][ix].argmax(1) == y[ix]).mean(), g, None, "choice_test"),
        "Choice, intenzioni mai viste": (lambda z, ix: (z["choice_test"][uix[ix]].argmax(1) == y[uix[ix]]).mean(), g[uix], None, "choice_test"),
        "Choice con le opzioni nella lingua della frase":
            (lambda z, ix: (z["choice_test_native"][ix].argmax(1) == y[ix]).mean(), g, None, "choice_test_native"),
        "Sì/no, test (AUROC)": (lambda z, ix: roc_auc_score(yt[ix], z["test"][ix]), gt, None, "test"),
        "Sì/no, test esteso mai visto (AUROC)": (lambda z, ix: roc_auc_score(ye[ix], z["ext"][ix]), ge, None, "ext"),
        "ECE calibrata delle Choice": (ece_choice, g, None, "choice_calib"),
        "ECE delle sì/no (Platt)": (ece_noul, gt, None, "calib"),
        "Divario Choice fra inglese e lingua peggiore": (gap, g, None, "choice_test"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obiettivi", required=True)
    ap.add_argument("--rif", required=True); ap.add_argument("--cand", required=True)
    for f in "abcd":
        ap.add_argument(f"--{f}_rif"); ap.add_argument(f"--{f}_cand")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ob = json.load(open(a.obiettivi))
    guardie = ob["misure_di_guardia"]
    # allow_pickle: eval_agent.py fino al 16 ottobre salvava le colonne di testo come oggetti; file nostri, locali
    carica = lambda p: np.load(p, allow_pickle=True) if p and os.path.exists(p) else None
    Z = {f: (carica(getattr(a, f"{f}_rif")), carica(getattr(a, f"{f}_cand"))) for f in "abcd"}
    righe = []

    def registra(m, v_rif, v_cand, vals, n, nota=""):
        """vals = differenze bootstrap ORIENTATE (positivo = meglio il candidato); IC riportato sulla differenza
        grezza candidato − riferimento."""
        su = m.get("verso", "su") != "giu"
        p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean())) if vals is not None else np.nan
        lo, hi = np.percentile(vals if su else -vals, [2.5, 97.5]) if vals is not None else (np.nan, np.nan)
        ok = v_rif == v_rif and v_cand == v_cand
        righe.append(dict(famiglia=m["famiglia"], misura=m["misura"], verso=m.get("verso", "su"),
                          partenza_file=m.get("valore_ft6"), rif=round(float(v_rif), 4) if ok else np.nan,
                          cand=round(float(v_cand), 4) if ok else np.nan,
                          diff=round(float(v_cand - v_rif), 4) if ok else np.nan,
                          ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4), p=p, n=n, nota=nota))

    # --- A
    A = misure_A(a, a.data); zr, zc = Z["a"]
    for m in [m for m in guardie if m["famiglia"] == "A"]:
        fn, grp, _, key = A[m["misura"]]
        if zr is None or zc is None or key not in zr.files or key not in zc.files:
            registra(m, np.nan, np.nan, None, m.get("n"), "non misurata: mancano i punteggi per esempio"); continue
        tutto = np.arange(len(grp))
        vr, vc = fn(zr, tutto), fn(zc, tutto)
        su = m.get("verso", "su") != "giu"
        vals = boot_appaiato(lambda ix: (fn(zc, ix) - fn(zr, ix)) * (1 if su else -1), grp, a.B)
        registra(m, vr, vc, vals, len(grp))
    # --- B
    zr, zc = Z["b"]
    VAR = {"Azioni, comando (AUROC)": "cmd|q_en", "Azioni, descrizione inglese (AUROC)": "desc_en",
           "Azioni, descrizione nella lingua (AUROC)": "desc_native"}
    for m in [m for m in guardie if m["famiglia"] == "B"]:
        if zr is None or zc is None or a.rif not in zr.files or a.cand not in zc.files:
            registra(m, np.nan, np.nan, None, m.get("n"), "non misurata: eval_agent.py non salvava i punteggi per esempio"); continue
        assert (zr["variant"] == zc["variant"]).all() and (zr["label"] == zc["label"]).all(), "item B diversi"
        mk = zr["variant"] == VAR[m["misura"]]; y = zr["label"][mk]; sr, sc = zr[a.rif][mk], zc[a.cand][mk]
        vals = boot_appaiato(lambda ix: roc_auc_score(y[ix], sc[ix]) - roc_auc_score(y[ix], sr[ix]), zr["page"][mk], a.B)
        registra(m, roc_auc_score(y, sr), roc_auc_score(y, sc), vals, int(mk.sum()))
    # --- C
    zr, zc = Z["c"]
    items = pd.DataFrame(read(f"{a.data}/ood/ood_v0.jsonl"))
    ch, no = items[items.task == "choice"], items[items.task == "noul"]
    for m in [m for m in guardie if m["famiglia"] == "C"]:
        l = "it" if "italiano" in m["misura"] else "en"
        tipo = "choice" if "Choice" in m["misura"] else "noul"
        kr, kc = f"{a.rif}_{tipo}_{l}", f"{a.cand}_{tipo}_{l}"
        if zr is None or zc is None or kr not in zr.files or kc not in zc.files:
            registra(m, np.nan, np.nan, None, m.get("n"), "non misurata"); continue
        sr, sc = zr[kr], zc[kc]
        if tipo == "choice":
            y = ch.gold_idx.values; g = ch.id.values
            st = lambda s, ix: (s[ix].argmax(1) == y[ix]).mean()
        else:
            y = no.label.values; g = no.id.values
            st = lambda s, ix: roc_auc_score(y[ix], s[ix])
        vals = boot_appaiato(lambda ix: st(sc, ix) - st(sr, ix), g, a.B)
        tutto = np.arange(len(y)); registra(m, st(sr, tutto), st(sc, tutto), vals, len(y))
    # --- D
    zr, zc = Z["d"]
    for m in [m for m in guardie if m["famiglia"] == "D"]:
        ds = JEV[m["misura"]]; kr, kc = f"{a.rif}|{ds}", f"{a.cand}|{ds}"
        if zr is None or zc is None or kr not in zr.files or kc not in zc.files:
            registra(m, np.nan, np.nan, None, m.get("n"), "non misurata"); continue
        y = zr[f"y|{ds}"]; assert (y == zc[f"y|{ds}"]).all(), ds
        pr, pc = zr[kr].astype(float), zc[kc].astype(float)
        st = (lambda p, ix: f1_pos(p[ix], y[ix])) if ds == "sms" else (lambda p, ix: (p[ix].argmax(1) == y[ix]).mean())
        vals = boot_appaiato(lambda ix: st(pc, ix) - st(pr, ix), np.arange(len(y)), a.B)
        tutto = np.arange(len(y)); registra(m, st(pr, tutto), st(pc, tutto), vals, len(y),
                                             "in dominio, letta a parte" if m["misura"] in IN_DOMINIO else "")

    df = pd.DataFrame(righe)
    # Holm dentro ogni famiglia, sulle misure misurate
    df["p_holm"] = np.nan
    for fam, d in df.groupby("famiglia"):
        d = d[d.p.notna()].sort_values("p"); k = len(d); run = 0.0
        for j, (i, r) in enumerate(d.iterrows()):
            run = max(run, min(1.0, (k - j) * r.p)); df.loc[i, "p_holm"] = run
    peggio = lambda r: (r["diff"] < 0) if r["verso"] != "giu" else (r["diff"] > 0)
    df["perdita"] = [bool(peggio(r)) for _, r in df.iterrows()]
    df["perdita_significativa"] = df.perdita & (df.p_holm < a.alpha)
    df["guadagno_significativo"] = (~df.perdita) & (df["diff"] != 0) & (df.p_holm < a.alpha)
    # tolleranze del giro G (copiate nel giro H), lette alla lettera
    giudizi = {}
    for fam, d in df.groupby("famiglia"):
        manca = d.p.isna().sum(); motivi = []
        if fam == "A":
            motivi += [f"perdita significativa: {r.misura}" for _, r in d[d.perdita_significativa].iterrows()]
        elif fam == "B":
            for _, r in d.iterrows():
                if "comando" in r.misura and r.perdita_significativa:
                    motivi.append(f"perdita significativa sul comando")
                elif "descrizione" in r.misura and (r.perdita_significativa or (r.perdita and abs(r["diff"]) > 0.02)):
                    motivi.append(f"{r.misura}: {r['diff']:+.4f}")
        elif fam == "C":
            motivi += [f"{r.misura}: {r['diff']:+.4f} (oltre 0,02)" for _, r in d.iterrows() if r.perdita and abs(r["diff"]) > 0.02]
            motivi += [f"{r.misura}: perdita significativa" for _, r in d.iterrows() if "sì/no" in r.misura and r.perdita_significativa]
        elif fam == "D":
            sig = d[d.perdita_significativa & ~d.misura.isin(IN_DOMINIO)]
            if len(sig) >= 2:
                motivi.append(f"{len(sig)} righe in perdita significativa: " + ", ".join(f"{r.misura} {r['diff']:+.3f}" for _, r in sig.iterrows()))
            elif len(sig) == 1 and abs(sig.iloc[0]["diff"]) > 0.03:
                motivi.append(f"una riga oltre 0,03: {sig.iloc[0].misura} {sig.iloc[0]['diff']:+.3f}")
        stato = "violata" if motivi else ("incompleta" if manca else "rispettata")
        giudizi[fam] = dict(stato=stato, motivi=motivi, non_misurate=int(manca), misure=len(d))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    riepilogo = dict(obiettivi=a.obiettivi, riferimento=a.rif, candidato=a.cand, ricampionamenti=a.B, alpha=a.alpha,
                     correzione="Holm dentro ogni famiglia", famiglie=giudizi,
                     regola_rispettata=all(v["stato"] == "rispettata" for v in giudizi.values()),
                     nota_banking77="Banking77 entra nella correzione di Holm della famiglia D ma non nel conteggio "
                                    "delle righe in perdita (in dominio, «letta a parte»)")
    json.dump(riepilogo, open(os.path.splitext(a.out)[0] + "_riepilogo.json", "w"), indent=1, ensure_ascii=False)
    pd.set_option("display.width", 200)
    print(df[["famiglia", "misura", "rif", "cand", "diff", "ci_lo", "ci_hi", "p_holm", "perdita_significativa"]].to_string(index=False))
    print(json.dumps(giudizi, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
