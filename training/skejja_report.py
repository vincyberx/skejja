"""Quality rows in the shared long format, using ONLY skejja_measure helpers for metrics/calibration."""
import numpy as np
import skejja_measure as fm

TRACK = "modelli_piccoli"


def _choice_block(lg, y, T):
    raw = fm.choice_metrics(fm.softmax(lg), y)
    cal = fm.choice_metrics(fm.softmax(lg, T), y)
    return {"accuracy": raw["accuracy"], "nll_raw": raw["nll"], "brier_raw": raw["brier"], "ece_raw": raw["ece"],
            "nll_cal": cal["nll"], "brier_cal": cal["brier"], "ece_cal": cal["ece"]}


def _noul_block(s, y, a, b):
    raw = fm.noul_metrics(fm.sigmoid(s), y)
    cal = fm.noul_metrics(fm.sigmoid(a * np.asarray(s) + b), y)
    return {"auroc": raw["auroc"], "accuracy": raw["accuracy"], "nll_raw": raw["nll"], "brier_raw": raw["brier"],
            "ece_raw": raw["ece"], "accuracy_cal": cal["accuracy"], "nll_cal": cal["nll"], "brier_cal": cal["brier"],
            "ece_cal": cal["ece"]}


def quality_rows(model, variant, runtime, threads, test_choice, test_noul, lg_choice, s_noul,
                 calib_choice, calib_noul, lg_choice_cal, s_noul_cal):
    """lg_choice: (N,8) raw logits for test_choice rows; s_noul: (N,) raw logits for test_noul rows.
    Calibration parameters fitted on the calib arrays only (pooled over languages)."""
    out = []
    T = fm.fit_temperature(lg_choice_cal, [r["gold_idx"] for r in calib_choice])
    a, b = fm.fit_platt(s_noul_cal, [r["label"] for r in calib_noul])
    base = dict(model=model, variant=variant, runtime=runtime, threads=threads)
    out += fm.rows(TRACK, **base, task="choice", split="calibration", lang="all", metrics={"temperature": T}, n=len(calib_choice))
    out += fm.rows(TRACK, **base, task="noul", split="calibration", lang="all", metrics={"platt_a": a, "platt_b": b}, n=len(calib_noul))
    lg_choice = np.asarray(lg_choice, float); s_noul = np.asarray(s_noul, float)
    for task, rows_, key in (("choice", test_choice, "gold"), ("noul", test_noul, "intent")):
        lang = np.array([r["lang"] for r in rows_]); uns = np.array([fm.is_unseen(r[key]) for r in rows_])
        for split, smask in (("all", np.ones(len(rows_), bool)), ("seen", ~uns), ("unseen", uns)):
            per = {}
            for L in fm.LANGS + ["all"]:
                m = smask & ((lang == L) if L != "all" else True)
                idx = np.where(m)[0]
                if task == "choice":
                    met = _choice_block(lg_choice[idx], np.array([rows_[i]["gold_idx"] for i in idx]), T)
                else:
                    met = _noul_block(s_noul[idx], np.array([rows_[i]["label"] for i in idx]), a, b)
                per[L] = (met, len(idx))
                out += fm.rows(TRACK, **base, task=task, split=split, lang=L, metrics=met, n=len(idx))
            macro = {k: float(np.mean([per[L][0][k] for L in fm.LANGS])) for k in per["en"][0]}
            out += fm.rows(TRACK, **base, task=task, split=split, lang="macro", metrics=macro,
                           n=int(sum(per[L][1] for L in fm.LANGS)))
    return out
