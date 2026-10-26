"""Torch-free inference for Skejja small models (onnxruntime + tokenizers + numpy only).

Input format (GLiClass uni-encoder, prompt_first): '<<LABEL>>opt1<<LABEL>>opt2...<<SEP>>text'
- Choice: K option descriptions (English) -> K logits -> softmax(logits / temperature)
- Noul:   question 'Does the user want to X?' is fed as the single label 'X' -> sigmoid(platt_a * logit + platt_b)
- Score:  the K level descriptions of an ordinal scale -> softmax / temperature_score -> expected value SUM p*value
Calibration parameters (fitted on calib_* only) are in calibration.json, keyed by .onnx filename.
Per una domanda nuova, Skejja.calibrate_question(domanda, [(testo, 0/1), ...]) adatta la calibrazione con esempi
etichettati (consigliati 32-64: results/fase2/metrics_fewshot.csv, docs/fase2_noul.md).
Weights are stored as ONNX external data (<model>.onnx.data, memory-mapped by onnxruntime): keep both files together.
Default model: the first .onnx in the bundle, except on ARM CPUs without dot-product instructions (no 'asimddp' in
/proc/cpuinfo, e.g. Raspberry Pi 3/4), where the dynamic int8 file (*.int8dyn.onnx) is used if present: there the
weight-only int8 kernel is about 2x slower (results/pi3/ANALISI.md). --int8dyn yes|no forces the choice.

usage: python skejja_infer.py <bundle_dir> "<text>" "opt1|opt2|..." [model.onnx] [--int8dyn auto|yes|no]   (choice)
       python skejja_infer.py <bundle_dir> "<text>" "Does the user want to X?" [model.onnx] [--int8dyn ...]  (noul)
       python skejja_infer.py <bundle_dir> "<text>" "liv0|liv1|..." --values 0,1,2,3,4                       (score)
"""
import json, os, platform, re, sys
import numpy as np
import onnxruntime as ort
if hasattr(ort, "disable_telemetry_events"):
    ort.disable_telemetry_events()  # keep everything local
import tokenizers

Q_RE = re.compile(r"^Does the user want to (.*)\?$")


def arm_without_dotprod(cpuinfo="/proc/cpuinfo"):
    """True on ARM CPUs whose /proc/cpuinfo lacks 'asimddp'; False elsewhere or when unknown (macOS, x86)."""
    if not platform.machine().lower().startswith(("arm", "aarch")):
        return False
    try:
        with open(cpuinfo) as f:
            return "asimddp" not in f.read().split()
    except OSError:
        return False


def pick_model(files, int8dyn="auto"):
    """Default .onnx of a bundle. int8dyn: 'auto' (dynamic int8 on ARM without dot-product), 'yes', 'no'."""
    dyn = [f for f in files if f.endswith(".int8dyn.onnx")]
    if int8dyn == "yes" and not dyn:
        raise FileNotFoundError("--int8dyn yes: nel bundle non c'è un file *.int8dyn.onnx")
    if dyn and (int8dyn == "yes" or (int8dyn == "auto" and arm_without_dotprod())):
        return dyn[0]
    return files[0]


class Skejja:
    def __init__(self, bundle_dir, threads=4, model_file=None, int8dyn="auto"):
        files = sorted(f for f in os.listdir(bundle_dir) if f.endswith(".onnx"))
        model_file = model_file or pick_model(files, int8dyn)
        self.model_file = model_file
        path = os.path.join(bundle_dir, model_file)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.tok = tokenizers.Tokenizer.from_file(os.path.join(bundle_dir, "tokenizer.json"))
        self.cal = json.load(open(os.path.join(bundle_dir, "calibration.json")))[model_file]

    def logits(self, text, labels):
        s = "".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>" + text
        ids = np.array([self.tok.encode(s).ids], dtype=np.int64)
        return self.sess.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids),
                                          "label_slots": np.zeros(len(labels), np.float32)})[0][0]

    def choice(self, text, options, calibrated=True):
        z = self.logits(text, options).astype(np.float64)
        z = z / (self.cal["temperature"] if calibrated else 1.0)
        p = np.exp(z - z.max()); p /= p.sum()
        return p

    def score(self, text, levels, values=None, calibrated=True):
        """Score: Choice sui livelli di una scala ordinale, più il valore atteso.

        `levels`: descrizioni dei livelli nell'ordine della scala; `values`: valore numerico di ciascun livello
        (predefinito 0, 1, ... K-1, cioè la scala 0-4 a cinque livelli di S1MB). Restituisce
        `(valore_atteso, probabilità)`: il valore atteso è Σ p·valore, cioè quello su cui S1MB misura l'errore.
        La temperatura è `temperature_score` di calibration.json se c'è (fittata solo sui `calib_*`), altrimenti
        quella delle Choice.
        """
        v = np.arange(len(levels), dtype=np.float64) if values is None else np.asarray(values, np.float64)
        if v.shape != (len(levels),):
            raise ValueError("score: servono tanti valori quanti sono i livelli")
        z = self.logits(text, levels).astype(np.float64)
        if calibrated:
            z = z / self.cal.get("temperature_score", self.cal["temperature"])
        p = np.exp(z - z.max()); p /= p.sum()
        return float((p * v).sum()), p

    def noul_logit(self, text, question):
        m = Q_RE.match(question.strip())
        return float(self.logits(text, [m.group(1) if m else question])[0])

    def noul(self, text, question, calibrated=True):
        z = self.noul_logit(text, question)
        if calibrated:
            a, b = self.question_cal.get(question.strip(), (self.cal["platt_a"], self.cal["platt_b"]))
            z = a * z + b
        return 1.0 / (1.0 + np.exp(-z))

    @property
    def question_cal(self):
        if not hasattr(self, "_qcal"):
            self._qcal = {}
        return self._qcal

    def calibrate_question(self, question, examples, sigma=None):
        """Calibrazione con pochi esempi per una domanda nuova. examples: lista di (testo, 0/1).
        Platt MAP ancorata alla calibrazione globale (vedi fit_platt_map); dopo la chiamata noul() usa i parametri
        di questa domanda. Restituisce (a, b)."""
        z = [self.noul_logit(t, question) for t, _ in examples]
        y = [int(l) for _, l in examples]
        ab = fit_platt_map(z, y, (self.cal["platt_a"], self.cal["platt_b"]),
                           sigma if sigma is not None else self.cal.get("fewshot_sigma", FEWSHOT_SIGMA))
        self.question_cal[question.strip()] = ab
        return ab


FEWSHOT_SIGMA = 0.5   # scelto sul set di sviluppo esterno (results/fase2/metrics_fewshot.csv); sovrascrivibile con "fewshot_sigma" in calibration.json


def fit_platt_map(z, y, prior, sigma, iters=50):
    """Platt (p = sigmoid(a*z + b)) stimata per massimo a posteriori con prior gaussiano isotropo N(prior, sigma^2)
    sui due parametri. Con pochi esempi resta vicina alla calibrazione globale `prior`; con molti esempi tende alla
    Platt della domanda. Newton-Raphson, solo numpy. z: logit grezzi del modello, y: etichette 0/1."""
    z = np.asarray(z, np.float64); y = np.asarray(y, np.float64)
    w0 = np.asarray(prior, np.float64); w = w0.copy(); lam = 1.0 / (sigma ** 2)
    X = np.stack([z, np.ones_like(z)], 1)

    def obj(w):   # log-loss stabile (logaddexp) + penalità del prior
        t = X @ w
        return float(np.sum(np.logaddexp(0.0, t) - y * t) + 0.5 * lam * np.sum((w - w0) ** 2))

    f = obj(w)
    for _ in range(iters):
        t = np.clip(X @ w, -40, 40); p = 1.0 / (1.0 + np.exp(-t))
        g = X.T @ (p - y) + lam * (w - w0)
        H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(2)
        step = np.linalg.solve(H, g)
        # Newton smorzato: si dimezza il passo finché l'obiettivo non scende (evita le divergenze)
        s = 1.0
        while s > 1e-6:
            wn = w - s * step; fn = obj(wn)
            if fn <= f:
                break
            s *= 0.5
        else:
            break
        done = np.abs(wn - w).max() < 1e-8
        w, f = wn, fn
        if done:
            break
    return float(w[0]), float(w[1])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Skejja inference (choice: 'opt1|opt2|...', noul: 'Does the user want to X?')")
    ap.add_argument("bundle_dir"); ap.add_argument("text"); ap.add_argument("arg")
    ap.add_argument("model_file", nargs="?", help="a specific .onnx in the bundle (overrides --int8dyn)")
    ap.add_argument("--int8dyn", choices=["auto", "yes", "no"], default="auto",
                    help="use the dynamic int8 file: auto = only on ARM without dot-product (default)")
    ap.add_argument("--values", default="", help="Score: valori dei livelli separati da virgola (es. 0,1,2,3,4)")
    a = ap.parse_args()
    f = Skejja(a.bundle_dir, model_file=a.model_file, int8dyn=a.int8dyn)
    print(f"modello: {f.model_file}", file=sys.stderr)
    text, arg = a.text, a.arg
    if a.values:
        levels = arg.split("|")
        atteso, p = f.score(text, levels, [float(x) for x in a.values.split(",")])
        print(f"valore atteso = {atteso:.3f}")
        for l, pi in zip(levels, p):
            print(f"{pi:.3f}  {l}")
    elif "|" in arg:
        opts = arg.split("|"); p = f.choice(text, opts)
        for o, pi in sorted(zip(opts, p), key=lambda x: -x[1]):
            print(f"{pi:.3f}  {o}")
    else:
        print(f"p_yes={f.noul(text, arg):.3f}")
