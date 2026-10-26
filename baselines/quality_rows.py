"""Build long-format quality rows (skejja_measure.METRIC_COLUMNS) from raw scores.

choice: logits (N,8) -> raw probs = softmax(logits); calibrated = softmax(logits / T), T fit on calib (all langs pooled).
noul:   score (N,) raw logit of yes -> raw p = sigmoid(score); calibrated p = sigmoid(a*score + b), (a,b) Platt on calib.
"""
import numpy as np
import skejja_measure as fm

TRACK = "baseline"


def _choice_m(p_raw, p_cal, y):
    r = fm.choice_metrics(p_raw, y)
    c = fm.choice_metrics(p_cal, y)
    return {"accuracy": r["accuracy"], "nll": r["nll"], "brier": r["brier"], "ece_raw": r["ece"],
            "nll_cal": c["nll"], "brier_cal": c["brier"], "ece_cal": c["ece"]}


def _noul_m(p_raw, p_cal, y):
    r = fm.noul_metrics(p_raw, y)
    c = fm.noul_metrics(p_cal, y)
    return {"auroc": r["auroc"], "accuracy": r["accuracy"], "brier": r["brier"], "nll": r["nll"], "ece_raw": r["ece"],
            "accuracy_cal": c["accuracy"], "brier_cal": c["brier"], "nll_cal": c["nll"], "ece_cal": c["ece"]}


def quality_rows(model, variant, runtime, threads, test_choice_rows, test_noul_rows,
                 z_test_c, z_cal_c, y_cal_c, s_test_n, s_cal_n, y_cal_n):
    out = []
    # ---- choice
    T = fm.fit_temperature(z_cal_c, y_cal_c)
    y = np.array([r["gold_idx"] for r in test_choice_rows])
    langs = np.array([r["lang"] for r in test_choice_rows])
    unseen = np.array([fm.is_unseen(r["gold"]) for r in test_choice_rows])
    p_raw, p_cal = fm.softmax(z_test_c), fm.softmax(z_test_c, T)
    out += fm.rows(TRACK, model, variant, runtime, threads, "choice", "all", "all", {"temperature": T}, len(y_cal_c))
    out += _split_lang_rows(model, variant, runtime, threads, "choice", langs, unseen,
                            lambda m: _choice_m(p_raw[m], p_cal[m], y[m]))
    # ---- noul
    a, b = fm.fit_platt(s_cal_n, y_cal_n)
    yn = np.array([r["label"] for r in test_noul_rows])
    langs = np.array([r["lang"] for r in test_noul_rows])
    unseen = np.array([fm.is_unseen(r["intent"]) for r in test_noul_rows])
    s = np.asarray(s_test_n, float)
    pr, pc = fm.sigmoid(s), fm.sigmoid(a * s + b)
    out += fm.rows(TRACK, model, variant, runtime, threads, "noul", "all", "all", {"platt_a": a, "platt_b": b}, len(y_cal_n))
    out += _split_lang_rows(model, variant, runtime, threads, "noul", langs, unseen,
                            lambda m: _noul_m(pr[m], pc[m], yn[m]))
    return out


def _split_lang_rows(model, variant, runtime, threads, task, langs, unseen, fn):
    out = []
    for split, smask in [("all", np.ones_like(unseen, bool)), ("seen", ~unseen), ("unseen", unseen)]:
        per = {}
        for lang in fm.LANGS:
            m = smask & (langs == lang)
            per[lang] = fn(m)
            out += fm.rows(TRACK, model, variant, runtime, threads, task, split, lang, per[lang], int(m.sum()))
        macro = {k: float(np.mean([per[l][k] for l in fm.LANGS])) for k in per[fm.LANGS[0]]}
        out += fm.rows(TRACK, model, variant, runtime, threads, task, split, "macro", macro, int(smask.sum()))
        out += fm.rows(TRACK, model, variant, runtime, threads, task, split, "all", fn(smask), int(smask.sum()))
    return out
