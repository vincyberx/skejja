"""Valutazione Noul (e Choice come controllo di regressione) con il protocollo di bench/skejja_measure.py.

Split Noul (`intent` = intenzione della domanda):
  all, seen_q (domanda su intenzione vista), unseen_q (domanda su intenzione mai vista = split "mai viste" della fase 1),
  unseen_q/neg_same e unseen_q/neg_other (positivi + un solo tipo di negativo), lo stesso per seen_q;
  unseen_ext (bench/data/test_noul_unseen_ext.jsonl, disgiunto dal test originale) e le sue scomposizioni.
Calibrazione: `platt_calib` = Platt globale su calib_noul.jsonl (protocollo fase 1); con --calib_ext anche
`platt_calib_ext` = Platt sul file indicato (domande su intenzioni escluse dall'addestramento, mai su MASSIVE test).
Intervalli al 95%: bootstrap per frase (le 5 lingue di una frase restano insieme), 1000 ripetizioni.

uso: python training/eval_noul.py --model releases/checkpoints/edge32k-ft --name edge32k-ft --variant torch-fp32 \
        --out results/fase2/metrics_edge32k-ft.csv
     (--model può essere anche un file .onnx di un bundle; per *.int8dyn.onnx usare --bs 1)
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "bench"))
import skejja_measure as fm  # noqa: E402
import skejja_glic as fg  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def read(path):
    return [json.loads(l) for l in open(path)]


class Backend:
    def __init__(self, model, threads, bs):
        self.bs = bs
        if model.endswith(".onnx"):
            import onnxruntime as ort
            ort.disable_telemetry_events()
            import tokenizers
            so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
            self.tok = tokenizers.Tokenizer.from_file(os.path.join(os.path.dirname(model), "tokenizer.json"))
            self.tok.enable_padding(pad_id=self.tok.token_to_id("<pad>"))
            self.kind = "onnx"
        else:
            import torch
            from gliclass import GLiClassModel
            from transformers import AutoTokenizer
            torch.set_num_threads(threads)
            # SKEJJA_DEVICE=cuda per valutare sulla GPU (workstation); TF32 spento, così i numeri restano fp32
            dev = os.environ.get("SKEJJA_DEVICE", "cpu")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.model = GLiClassModel.from_pretrained(model, dtype=torch.float32).to(dev).eval()
            self.tok = AutoTokenizer.from_pretrained(model)
            self.kind = "torch"

    def __call__(self, inputs, K):
        if self.kind == "torch":
            return fg.score_torch(self.model, self.tok, inputs, K, batch_size=self.bs)
        order = np.argsort([len(s) for s in inputs]); out = np.zeros((len(inputs), K), np.float32)
        for i in range(0, len(order), self.bs):
            idx = order[i:i + self.bs]; enc = self.tok.encode_batch([inputs[j] for j in idx])
            ids = np.array([e.ids for e in enc], np.int64); am = np.array([e.attention_mask for e in enc], np.int64)
            out[idx] = self.sess.run(["logits"], {"input_ids": ids, "attention_mask": am,
                                                  "label_slots": np.zeros(K, np.float32)})[0][:, :K]
        return out


def noul_frame(rows, choice_rows, scores):
    d = pd.DataFrame(rows)
    d["neg_type"] = fm.noul_neg_type(rows, choice_rows)
    d["s"] = scores
    d["q_unseen"] = d.intent.map(fm.is_unseen)
    return d


def noul_rows(d, p, split, lang, ci, meta):
    out = []
    m = fm.noul_metrics(p, d.label.values)
    if ci and len(d) > 0:
        grp = d.id.values; y = d.label.values; s = d.s.values; pp = np.asarray(p)
        lo, hi = fm.bootstrap_ci(lambda ix: roc_auc_score(y[ix], s[ix]), grp)
        m["auroc_ci_lo"], m["auroc_ci_hi"] = lo, hi
        def _ece(ix):
            pr = (pp[ix] >= 0.5).astype(int); conf = np.where(pr == 1, pp[ix], 1 - pp[ix])
            return fm.ece(conf, pr == y[ix])
        lo, hi = fm.bootstrap_ci(_ece, grp)
        m["ece_ci_lo"], m["ece_ci_hi"] = lo, hi
    m["mean_p_pos"] = float(np.mean(np.asarray(p)[d.label.values == 1])) if (d.label == 1).any() else float("nan")
    m["mean_p_neg"] = float(np.mean(np.asarray(p)[d.label.values == 0])) if (d.label == 0).any() else float("nan")
    out += fm.rows(meta["track"], meta["model"], meta["variant"], meta["runtime"], meta["threads"], "noul",
                   split, lang, m, len(d))
    return out


def native_rows(be, D, meta, variant):
    """Opzioni e domande nella lingua della frase (traduzioni OPUS-MT delle descrizioni del benchmark; per l'inglese
    la descrizione originale). Calibrazione rifittata sui file calib_* con le stesse traduzioni."""
    mt = json.load(open(f"{D}/intent_descriptions_mt.json"))
    inv = {v["en"]: k for k, v in mt.items()}
    nat = lambda en_desc, lang: mt[inv[en_desc]][lang]
    out = []; mt_ = dict(meta, variant=f"{variant}|native")
    cc, tc = read(f"{D}/calib_choice.jsonl"), read(f"{D}/test_choice.jsonl")
    cin = lambda rows: [fg.fmt(r["text"], [nat(o, r["lang"]) for o in r["option_desc"]]) for r in rows]
    lc, lt = be(cin(cc), 8), be(cin(tc), 8)
    raw_nat = dict(choice_calib_native=lc, choice_test_native=lt)
    T = fm.fit_temperature(lc, [r["gold_idx"] for r in cc])
    y = np.array([r["gold_idx"] for r in tc]); uns = np.array([fm.is_unseen(r["gold"]) for r in tc])
    lang = np.array([r["lang"] for r in tc])
    for nm, m in (("all", np.ones(len(y), bool)), ("seen", ~uns), ("unseen", uns)):
        out += fm.rows(mt_["track"], mt_["model"], mt_["variant"] + "|temp_calib", mt_["runtime"], mt_["threads"], "choice",
                       nm, "pooled", fm.choice_metrics(fm.softmax(lt[m], T), y[m]), int(m.sum()))
        for l in fm.LANGS:
            ml = m & (lang == l)
            out += fm.rows(mt_["track"], mt_["model"], mt_["variant"] + "|temp_calib", mt_["runtime"], mt_["threads"], "choice",
                           nm, l, fm.choice_metrics(fm.softmax(lt[ml], T), y[ml]), int(ml.sum()))
    nin = lambda rows: [fg.fmt(r["text"], [nat(fg.noul_label(r["question"]), r["lang"])]) for r in rows]
    cal_n, te_n, ext_n = read(f"{D}/calib_noul.jsonl"), read(f"{D}/test_noul.jsonl"), read(f"{D}/test_noul_unseen_ext.jsonl")
    cal = noul_frame(cal_n, read(f"{D}/calib_choice.jsonl"), be(nin(cal_n), 1)[:, 0])
    te = noul_frame(te_n, tc, be(nin(te_n), 1)[:, 0]); ext = noul_frame(ext_n, None, be(nin(ext_n), 1)[:, 0])
    raw_nat.update(test_native=te.s.values, ext_native=ext.s.values)
    pa, pb = fm.fit_platt(cal.s, cal.label)
    P = lambda d: fm.sigmoid(pa * d.s.values + pb)
    mtn = dict(mt_, variant=mt_["variant"] + "|platt_calib")
    for nm, d in (("all", te), ("seen_q", te[~te.q_unseen]), ("unseen_q", te[te.q_unseen]), ("unseen_ext", ext)):
        out += noul_rows(d, P(d), nm, "pooled", True, mtn)
        if nm != "all":
            dd = d[d.neg_type.isin(["pos", "neg_same"])]
            out += noul_rows(dd, P(dd), f"{nm}/neg_same", "pooled", True, mtn)
        for l in fm.LANGS:
            dl = d[d.lang == l]
            out += noul_rows(dl, P(dl), nm, l, False, mtn)
    return out, raw_nat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", default="torch-fp32")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "bench", "data"))
    ap.add_argument("--calib_ext", default="", help="file Noul di calibrazione su intenzioni escluse (opzionale)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--no_choice", action="store_true")
    ap.add_argument("--track", default="fase2-noul")
    ap.add_argument("--native", action="store_true",
                    help="anche la variante con opzioni e domande nella lingua della frase (bench/data/intent_descriptions_mt.json)")
    a = ap.parse_args()
    D = a.data
    be = Backend(a.model, a.threads, a.bs)
    meta = dict(track=a.track, model=a.name, variant=a.variant, runtime=be.kind, threads=a.threads)
    cal_n, te_n, ext_n = read(f"{D}/calib_noul.jsonl"), read(f"{D}/test_noul.jsonl"), read(f"{D}/test_noul_unseen_ext.jsonl")
    cal_c, te_c = read(f"{D}/calib_choice.jsonl"), read(f"{D}/test_choice.jsonl")
    sc = lambda rows: be(fg.noul_inputs(rows), 1)[:, 0]
    cal = noul_frame(cal_n, cal_c, sc(cal_n)); te = noul_frame(te_n, te_c, sc(te_n)); ext = noul_frame(ext_n, None, sc(ext_n))
    cals = {"platt_calib": fm.fit_platt(cal.s, cal.label)}
    raw = dict(calib=cal.s.values, test=te.s.values, ext=ext.s.values)
    if a.calib_ext:
        ce_n = read(a.calib_ext); ce = noul_frame(ce_n, None, sc(ce_n)); raw["calib_ext"] = ce.s.values
        cals["platt_calib_ext"] = fm.fit_platt(ce.s, ce.label)
    out = []
    for cname, (pa, pb) in cals.items():
        out += fm.rows(meta["track"], meta["model"], meta["variant"], meta["runtime"], meta["threads"], "noul",
                       "calibration", cname, {"platt_a": pa, "platt_b": pb}, len(cal))
        mt = dict(meta, variant=f"{a.variant}|{cname}")
        P = lambda d: fm.sigmoid(pa * d.s.values + pb)
        splits = [("all", te), ("seen_q", te[~te.q_unseen]), ("unseen_q", te[te.q_unseen]), ("unseen_ext", ext)]
        for nm, d in splits:
            out += noul_rows(d, P(d), nm, "pooled", True, mt)
            if nm != "all":
                for nt in ("neg_same", "neg_other"):
                    dd = d[d.neg_type.isin(["pos", nt])]
                    out += noul_rows(dd, P(dd), f"{nm}/{nt}", "pooled", True, mt)
            for l in fm.LANGS:
                dl = d[d.lang == l]
                out += noul_rows(dl, P(dl), nm, l, False, mt)
    if not a.no_choice:
        lc = be(fg.choice_inputs(cal_c), 8); lt = be(fg.choice_inputs(te_c), 8)
        T = fm.fit_temperature(lc, [r["gold_idx"] for r in cal_c])
        y = np.array([r["gold_idx"] for r in te_c]); uns = np.array([fm.is_unseen(r["gold"]) for r in te_c])
        lang = np.array([r["lang"] for r in te_c])
        mt = dict(meta, variant=f"{a.variant}|temp_calib")
        out += fm.rows(meta["track"], meta["model"], meta["variant"], meta["runtime"], meta["threads"], "choice",
                       "calibration", "temp_calib", {"temperature": T}, len(cal_c))
        for nm, m in (("all", np.ones(len(y), bool)), ("seen", ~uns), ("unseen", uns)):
            out += fm.rows(mt["track"], mt["model"], mt["variant"], mt["runtime"], mt["threads"], "choice", nm,
                           "pooled", fm.choice_metrics(fm.softmax(lt[m], T), y[m]), int(m.sum()))
            for l in fm.LANGS:
                ml = m & (lang == l)
                out += fm.rows(mt["track"], mt["model"], mt["variant"], mt["runtime"], mt["threads"], "choice", nm, l,
                               fm.choice_metrics(fm.softmax(lt[ml], T), y[ml]), int(ml.sum()))
        raw.update(choice_calib=lc, choice_test=lt)
    if a.native:
        righe_nat, raw_nat = native_rows(be, D, meta, a.variant)
        out += righe_nat; raw.update(raw_nat)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df = pd.DataFrame(out, columns=fm.METRIC_COLUMNS)
    df.to_csv(a.out, index=False)
    np.savez_compressed(os.path.splitext(a.out)[0] + "_scores.npz", **raw)
    # logit per l'esperimento di calibrazione per domanda (training/fewshot_calib.py)
    if os.path.exists(f"{D}/dev_noul_ext.jsonl") and os.path.exists(f"{D}/calib_noul_fewshot.jsonl"):
        suffix = "" if a.variant == "torch-fp32" else f"_{a.variant}"   # i punteggi PyTorch non vanno sovrascritti
        np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(a.out)), f"scores_fewshot_{a.name}{suffix}.npz"),
                            dev_ext=sc(read(f"{D}/dev_noul_ext.jsonl")), fewshot=sc(read(f"{D}/calib_noul_fewshot.jsonl")))
    key = df[(df.lang == "pooled") & df.metric.isin(["auroc", "ece", "accuracy"])]
    print(key.pivot_table(index=["variant", "task", "split"], columns="metric", values="value").round(3).to_string())


if __name__ == "__main__":
    main()
