"""Julia 1 (SupersonicLabs, Apache-2.0) sul benchmark Skejja, con lo stesso protocollo e le stesse righe CSV di
training/eval_noul.py (quindi confrontabile riga per riga con results/fase2/metrics_edge32k-ft6.csv).

Modello preso com'è, zero-shot: nessun addestramento. Si fittano solo i calibratori post-hoc sui file calib_*
(temperatura per le Choice, Platt per le sì/no), come per ft6 e per le baseline della fase 1.

Come viene interrogato (API legacy di julia, `engine.logits(rows)`, stessa codifica di `predict`):
- Choice: domanda "What does the user want to do?" (la stessa data a Laya), opzioni = le 8 `option_desc` inglesi;
  con --native le stesse descrizioni tradotte (bench/data/intent_descriptions_mt.json), come la variante native di ft6.
- Noul: domanda = il campo `question`, opzioni ["false", "true"] (quello che fa l'API tipizzata quando la domanda
  non ha criteri); punteggio = z_true - z_false.

Esposizione: il provenance.json del modello riporta una validazione su MASSIVE (scenari, 52 lingue); i dati di
addestramento non sono pubblici, quindi non si può escludere che frasi del test MASSIVE siano state viste.

uso: JULIA_DIR=<snapshot di SupersonicLabs/Julia-1> python baselines/eval_julia.py \
        --out results/fase3/metrics_julia-1.csv --native
     (serve il pacchetto `julia` del repository del modello sul PYTHONPATH o in JULIA_DIR; transformers 5.0.x)
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training")); sys.path.insert(0, os.path.join(HERE, "..", "bench"))
import skejja_measure as fm  # noqa: E402
from eval_noul import read, noul_frame, noul_rows  # noqa: E402

JULIA_REPO, JULIA_REV = "SupersonicLabs/Julia-1", "a85b127321d580d65176c89ced8273f305745d85"
JULIA_SHA256 = "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72"
CHOICE_Q = "What does the user want to do?"


class Julia:
    def __init__(self, path, threads, bs):
        # il runtime di julia rimette i thread a JULIA_CPU_THREADS (predefinito 4) quando carica il modello
        os.environ["JULIA_CPU_THREADS"] = str(threads)
        sys.path.insert(0, path)
        from julia import load_model
        self.engine = load_model(path, device="cpu", strict_encoding=True, max_length=1024, head_length=512)
        self.bs = bs

    def _logits(self, rows):
        order = np.argsort([len(r["state"]) for r in rows]); out = [None] * len(rows)
        for i in range(0, len(order), self.bs):
            idx = order[i:i + self.bs]
            for j, z in zip(idx, self.engine.logits([rows[k] for k in idx])):
                out[j] = z
        return np.array(out, np.float64)

    def choice(self, rows, opts=lambda r: r["option_desc"]):
        return self._logits([dict(state=r["text"], question=CHOICE_Q, options=list(opts(r)), type="choice")
                             for r in rows])

    def noul(self, rows):
        z = self._logits([dict(state=r["text"], question=r["question"], options=["false", "true"], type="noul")
                          for r in rows])
        return z[:, 1] - z[:, 0]


def choice_rows(meta, variant, lc, cal_c, lt, te_c):
    out = []
    T = fm.fit_temperature(lc, [r["gold_idx"] for r in cal_c])
    y = np.array([r["gold_idx"] for r in te_c]); uns = np.array([fm.is_unseen(r["gold"]) for r in te_c])
    lang = np.array([r["lang"] for r in te_c])
    out += fm.rows(meta["track"], meta["model"], variant, meta["runtime"], meta["threads"], "choice",
                   "calibration", "temp_calib", {"temperature": T}, len(cal_c))
    v = f"{variant}|temp_calib"
    for nm, m in (("all", np.ones(len(y), bool)), ("seen", ~uns), ("unseen", uns)):
        met = fm.choice_metrics(fm.softmax(lt[m], T), y[m])
        met["ece_raw"] = fm.choice_metrics(fm.softmax(lt[m]), y[m])["ece"]
        out += fm.rows(meta["track"], meta["model"], v, meta["runtime"], meta["threads"], "choice", nm, "pooled",
                       met, int(m.sum()))
        for l in fm.LANGS:
            ml = m & (lang == l)
            out += fm.rows(meta["track"], meta["model"], v, meta["runtime"], meta["threads"], "choice", nm, l,
                           fm.choice_metrics(fm.softmax(lt[ml], T), y[ml]), int(ml.sum()))
    return out


NOUL_OPZIONI = {"literal": ["false", "true"], "no_yes": ["no", "yes"],
                "user_wants": ["No, the user does not want this.", "Yes, the user wants this."],
                "laya_default": ["no, the statement does not hold", "yes, the statement holds"]}


def ablazioni_noul(be, te_n, D, meta):
    """Diagnosi: AUROC delle sì/no del test con opzioni diverse da false/true, e la stessa domanda posta come Choice
    a due opzioni ("something else" contro la descrizione dell'intenzione). L'AUROC non dipende dalla calibrazione."""
    from sklearn.metrics import roc_auc_score
    y = np.array([r["label"] for r in te_n]); lang = np.array([r["lang"] for r in te_n]); out = []
    desc = json.load(open(f"{D}/intent_descriptions.json"))
    var = {k: be._logits([dict(state=r["text"], question=r["question"], options=o, type="noul") for r in te_n])
           for k, o in NOUL_OPZIONI.items()}
    var["choice2"] = be._logits([dict(state=r["text"], question=CHOICE_Q, options=["something else", desc[r["intent"]]],
                                      type="choice") for r in te_n])
    for k, z in var.items():
        s = z[:, 1] - z[:, 0]
        out += fm.rows(meta["track"], meta["model"], f"{meta['variant']}|ablazione_{k}", meta["runtime"], meta["threads"],
                       "noul", "all", "pooled", {"auroc": float(roc_auc_score(y, s))}, len(y))
        for l in fm.LANGS:
            m = lang == l
            out += fm.rows(meta["track"], meta["model"], f"{meta['variant']}|ablazione_{k}", meta["runtime"],
                           meta["threads"], "noul", "all", l, {"auroc": float(roc_auc_score(y[m], s[m]))}, int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--julia", default=os.environ.get("JULIA_DIR", ""))
    ap.add_argument("--data", default=os.path.join(HERE, "..", "bench", "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--ablazioni", action="store_true",
                    help="AUROC delle sì/no del test con altre formulazioni (solo diagnosi, nessuna scelta sul test)")
    a = ap.parse_args()
    D = a.data
    be = Julia(a.julia, a.threads, a.bs)
    meta = dict(track="fase3-confronto", model="julia-1", variant="torch-fp32", runtime="torch", threads=a.threads)
    t0 = time.time()
    cal_n, te_n, ext_n = read(f"{D}/calib_noul.jsonl"), read(f"{D}/test_noul.jsonl"), read(f"{D}/test_noul_unseen_ext.jsonl")
    ce_n = read(f"{D}/calib_noul_ext.jsonl")
    cal_c, te_c = read(f"{D}/calib_choice.jsonl"), read(f"{D}/test_choice.jsonl")
    cal = noul_frame(cal_n, cal_c, be.noul(cal_n)); te = noul_frame(te_n, te_c, be.noul(te_n))
    ext = noul_frame(ext_n, None, be.noul(ext_n)); ce = noul_frame(ce_n, None, be.noul(ce_n))
    raw = dict(calib=cal.s.values, test=te.s.values, ext=ext.s.values, calib_ext=ce.s.values)
    out = []
    for cname, (pa, pb) in {"platt_calib": fm.fit_platt(cal.s, cal.label),
                            "platt_calib_ext": fm.fit_platt(ce.s, ce.label)}.items():
        out += fm.rows(meta["track"], meta["model"], meta["variant"], meta["runtime"], meta["threads"], "noul",
                       "calibration", cname, {"platt_a": pa, "platt_b": pb}, len(cal))
        mt = dict(meta, variant=f"{meta['variant']}|{cname}")
        P = lambda d: fm.sigmoid(pa * d.s.values + pb)
        for nm, d in (("all", te), ("seen_q", te[~te.q_unseen]), ("unseen_q", te[te.q_unseen]), ("unseen_ext", ext)):
            out += noul_rows(d, P(d), nm, "pooled", True, mt)
            if nm != "all":
                for nt in ("neg_same", "neg_other"):
                    dd = d[d.neg_type.isin(["pos", nt])]
                    out += noul_rows(dd, P(dd), f"{nm}/{nt}", "pooled", True, mt)
            for l in fm.LANGS:
                dl = d[d.lang == l]
                out += noul_rows(dl, P(dl), nm, l, False, mt)
    lc, lt = be.choice(cal_c), be.choice(te_c)
    out += choice_rows(meta, meta["variant"], lc, cal_c, lt, te_c)
    raw.update(choice_calib=lc, choice_test=lt)
    if a.native:
        mt = json.load(open(f"{D}/intent_descriptions_mt.json"))
        inv = {v["en"]: k for k, v in mt.items()}
        nat = lambda r: [mt[inv[o]][r["lang"]] for o in r["option_desc"]]
        lcn, ltn = be.choice(cal_c, nat), be.choice(te_c, nat)
        out += choice_rows(meta, f"{meta['variant']}|native", lcn, cal_c, ltn, te_c)
        raw.update(choice_calib_native=lcn, choice_test_native=ltn)
    if a.ablazioni:
        out += ablazioni_noul(be, te_n, D, meta)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df = pd.DataFrame(out, columns=fm.METRIC_COLUMNS)
    df.to_csv(a.out, index=False)
    np.savez_compressed(os.path.splitext(a.out)[0] + "_scores.npz", **raw)
    print(f"tempo {time.time() - t0:.0f} s; {JULIA_REPO}@{JULIA_REV}")
    key = df[(df.lang == "pooled") & df.metric.isin(["auroc", "ece", "accuracy"])]
    print(key.pivot_table(index=["variant", "task", "split"], columns="metric", values="value").round(3).to_string())


if __name__ == "__main__":
    main()
