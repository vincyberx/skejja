"""Endpoint System One (`POST /v1/systemone`) sopra il runtime di Skejja, senza dipendenze web.

Contratto, dalla documentazione pubblica di Jev (docs.typesafe.ai: primitives, confidence) e dall'harness S1MB
(MIT, adattatore `typesafe.py`):

    richiesta  {"model"?: str, "state": testo | JSON,
                "questions": {id: {"type": "choice" | "noul" | "score",
                                   "instructions"?: testo | oggetto | lista,
                                   "criteria"?: ...}}}
    risposta   {"model": str, "answers": {id: risposta}, "usage": {"input_tokens": int, "output_tokens": 0}}

- choice: `criteria` è {nome_opzione: descrizione} (da 2 a 255 opzioni); risposta `choice` (l'opzione più
  probabile), `probabilities` per nome, `confidence` = (p_max − 1/n) / (1 − 1/n).
- noul: `criteria` facoltativo {"true": ..., "false": ...}; risposta `noul`, la probabilità del sì.
- score: `criteria` è la lista dei livelli dal più basso al più alto (da 2 a 10); risposta `score` = Σ p·livello,
  `probabilities` con chiavi "0", "1", ..., `legend` e `confidence` = max(0, 1 − Σ p_i |i − m| / MAD_unif).

Skejja ha un solo ingresso, `<<LABEL>>opzione…<<SEP>>testo`, quindi la resa è fissata qui una volta per tutte, come
nell'adattatore S1MB (bench/s1mb_adapter/skejja.py): le descrizioni dei criteri diventano le etichette nell'ordine
dato, le istruzioni vanno nel testo prima dello stato. Il nome di un'opzione arriva al modello solo se la sua
descrizione manca. Gli ingressi più lunghi di `max_tokens` si accorciano testa e coda (60/40) con un segno esplicito,
e la risposta elenca le domande accorciate nel campo `skejja.truncated`, che i client di Jev ignorano.

In più rispetto a Jev, la calibrazione per domanda delle noul: `calibrate()` riceve una domanda e i suoi esempi
etichettati, stima una Platt MAP ancorata alla calibrazione globale (`fit_platt_map`, come
`Skejja.calibrate_question`) e la ricorda per quel modello; da lì ogni noul con la stessa domanda la usa.
`skejja.calibration` dice per ogni noul se la probabilità è calibrata sulla domanda (`question`) o solo con la
calibrazione globale del bundle (`global`). Con `calibrations_path` le calibrazioni si salvano in un file JSON.
"""
import hashlib
import json
import os
import threading

import numpy as np
from skejja_infer import FEWSHOT_SIGMA, fit_platt_map

ELLIPSIS = "\n[…]\n"
TYPES = ("choice", "noul", "score")
MAX_CHOICE_OPTIONS = 255   # come Jev
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10   # come Jev
DEFAULT_MAX_TOKENS = 512   # sul Pi 4 la RAM di picco resta sotto i 300 MB (results/fase2/costo_lunghezza_opzioni_pi4.csv)
DEFAULT_MAX_QUESTIONS = 256
MIN_CALIBRATION_EXAMPLES = 8   # consigliati 32-64: results/fase2/metrics_fewshot.csv, docs/fase2_noul.md
MAX_CALIBRATION_EXAMPLES = 256
MAX_CALIBRATIONS = 1000        # calibrazioni ricordate per server: memoria e file restano limitati
MAX_TEXT_CHARS = 200_000       # stato e istruzioni si tokenizzano interi prima dell'accorciamento


class RequestError(ValueError):
    """Richiesta non valida: il server la restituisce come HTTP 400 con `param` = il campo sbagliato."""

    def __init__(self, message, param=None):
        super().__init__(message)
        self.message = message
        self.param = param


def as_text(value):
    """Testo così com'è; oggetti e liste in JSON compatto, con le chiavi nell'ordine dato."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def choice_confidence(p):
    n = len(p)
    return float((max(p) - 1.0 / n) / (1.0 - 1.0 / n))


def score_confidence(p):
    p = np.asarray(p, np.float64)
    n = len(p)
    levels = np.arange(n)
    m = int(np.argmax(p))
    mad_unif = float(np.mean(np.abs(levels - (n - 1) / 2.0)))
    return float(max(0.0, 1.0 - float(np.sum(p * np.abs(levels - m))) / mad_unif))


class SystemOne:
    """Risponde alle richieste System One con un modello Skejja (`skejja_infer.Skejja` o un oggetto con la stessa
    interfaccia: `tok`, `choice`, `noul`, `score`). Senza stato fra le richieste, quindi usabile da più thread."""

    def __init__(self, model, name="skejja", max_tokens=DEFAULT_MAX_TOKENS, max_questions=DEFAULT_MAX_QUESTIONS,
                 calibrations_path=None):
        if max_tokens < 64:
            raise ValueError("max_tokens deve lasciare spazio a criteri e testo (almeno 64)")
        self.model = model
        self.name = name
        self.max_tokens = max_tokens
        self.max_questions = max_questions
        self.calibrations_path = calibrations_path
        self._lock = threading.Lock()
        # chiave: [modello, etichetta, istruzioni nel testo]; la scala dei logit cambia da un .onnx all'altro,
        # quindi una calibrazione vale solo per il modello con cui è stata stimata
        self._calibrations = {}
        if calibrations_path and os.path.exists(calibrations_path):
            with open(calibrations_path) as f:
                self._calibrations = json.load(f).get("calibrations", {})

    # ---------- validazione ----------
    def _question(self, qid, q, where=None):
        where = where or f"questions.{qid}"
        if not isinstance(q, dict):
            raise RequestError("ogni domanda deve essere un oggetto", where)
        kind = q.get("type")
        if len(as_text(q.get("instructions"))) + len(as_text(q.get("criteria"))) > MAX_TEXT_CHARS:
            raise RequestError(f"istruzioni e criteri superano {MAX_TEXT_CHARS} caratteri", where)
        if kind not in TYPES:
            raise RequestError(f"type deve essere uno fra {', '.join(TYPES)}", f"{where}.type")
        instructions, criteria = q.get("instructions"), q.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
                raise RequestError(f"criteria deve essere un oggetto con da 2 a {MAX_CHOICE_OPTIONS} opzioni",
                                   f"{where}.criteria")
            if any(not isinstance(k, str) or not k for k in criteria):
                raise RequestError("i nomi delle opzioni devono essere testi non vuoti", f"{where}.criteria")
        elif kind == "score":
            if not isinstance(criteria, list) or not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
                raise RequestError(f"criteria deve essere una lista con da {MIN_SCORE_LEVELS} a {MAX_SCORE_LEVELS}"
                                   " livelli, dal più basso al più alto", f"{where}.criteria")
        else:
            if criteria is not None and (not isinstance(criteria, dict) or "true" not in criteria):
                raise RequestError('criteria della noul deve essere un oggetto con la chiave "true"',
                                   f"{where}.criteria")
            if criteria is None and not as_text(instructions).strip():
                raise RequestError("una noul senza criteria deve avere instructions", f"{where}.instructions")
        return kind, instructions, criteria

    def validate(self, request):
        if not isinstance(request, dict):
            raise RequestError("il corpo della richiesta deve essere un oggetto JSON")
        if "state" not in request:
            raise RequestError("manca state", "state")
        if len(as_text(request["state"])) > MAX_TEXT_CHARS:
            raise RequestError(f"state supera {MAX_TEXT_CHARS} caratteri", "state")
        questions = request.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise RequestError("questions deve essere un oggetto non vuoto {id: domanda}", "questions")
        if len(questions) > self.max_questions:
            raise RequestError(f"al massimo {self.max_questions} domande per richiesta", "questions")
        return {qid: self._question(qid, q) for qid, q in questions.items()}

    # ---------- resa, fissata prima ----------
    def _tokens(self, text):
        return len(self.model.tok.encode(text).ids)

    def _shorten(self, text, budget):
        """Testa 60% e coda 40% dei token ammessi, con il segno di taglio dell'harness S1MB.

        Il segno e la ricodifica ai bordi aggiungono qualche token: si ricontrolla la lunghezza e si stringe finché
        il testo accorciato sta davvero nel budget."""
        ids = self.model.tok.encode(text).ids
        if len(ids) <= budget:
            return text, False
        keep = budget - self._tokens(ELLIPSIS)
        while True:
            head = (keep * 3) // 5
            out = self.model.tok.decode(ids[:head]) + ELLIPSIS + self.model.tok.decode(ids[len(ids) - (keep - head):])
            excess = self._tokens(out) - budget
            if excess <= 0 or keep <= 8:
                return out, True
            keep -= excess

    def render(self, qid, kind, instructions, criteria, state):
        """Restituisce (testo, etichette, token totali, accorciato)."""
        head = as_text(instructions)
        if kind == "choice":
            labels = [as_text(d) or name.replace("_", " ") for name, d in criteria.items()]
        elif kind == "score":
            labels = [as_text(d) for d in criteria]
        elif criteria is not None:
            labels = [as_text(criteria["true"])]
        else:
            # senza criteri la domanda stessa è l'etichetta (come Skejja.noul) e nel testo resta solo lo stato
            labels, head = [head], ""
        body = f"{head}\n\n{state}" if head else state
        prefix = "".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>"
        label_tokens = self._tokens(prefix)
        if label_tokens > self.max_tokens - 32:
            raise RequestError(f"i criteri da soli superano il limite di {self.max_tokens} token",
                               f"questions.{qid}.criteria")
        text, cut = self._shorten(body, self.max_tokens - label_tokens)
        key = json.dumps([self.name, labels[0], head], ensure_ascii=False) if kind == "noul" else None
        # token della sequenza intera, come la vede il modello (Skejja.logits)
        return text, labels, self._tokens(prefix + text), cut, key

    # ---------- risposta ----------
    def answer(self, kind, text, labels, criteria, key=None):
        if kind == "choice":
            p = [float(x) for x in self.model.choice(text, labels)]
            names = list(criteria)
            return {"type": "choice", "choice": names[int(np.argmax(p))],
                    "probabilities": dict(zip(names, p)), "confidence": choice_confidence(p)}
        if kind == "noul":
            cal = self._calibrations.get(key)
            if cal is None:
                return {"type": "noul", "noul": float(self.model.noul(text, labels[0]))}
            z = cal["a"] * self.model.noul_logit(text, labels[0]) + cal["b"]
            return {"type": "noul", "noul": float(1.0 / (1.0 + np.exp(-z)))}
        expected, p = self.model.score(text, labels, list(range(len(labels))))
        p = [float(x) for x in p]
        return {"type": "score", "score": float(expected),
                "probabilities": {str(i): v for i, v in enumerate(p)},
                "legend": {str(i): c for i, c in enumerate(criteria)},
                "confidence": score_confidence(p)}

    def run(self, request):
        questions = self.validate(request)
        state = as_text(request["state"])
        answers, truncated, calibration, input_tokens = {}, [], {}, 0
        for qid, (kind, instructions, criteria) in questions.items():
            text, labels, n_tokens, cut, key = self.render(qid, kind, instructions, criteria, state)
            answers[qid] = self.answer(kind, text, labels, criteria, key)
            input_tokens += n_tokens
            if cut:
                truncated.append(qid)
            if kind == "noul":
                calibration[qid] = "question" if key in self._calibrations else "global"
        return {"model": self.name, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                "skejja": {"max_tokens": self.max_tokens, "truncated": truncated, "calibration": calibration}}

    # ---------- calibrazione per domanda (in più rispetto a Jev) ----------
    def calibrate(self, request):
        """{"question": <una noul>, "examples": [{"state": ..., "answer": true/false}, ...]} -> parametri stimati."""
        if not isinstance(request, dict):
            raise RequestError("il corpo della richiesta deve essere un oggetto JSON")
        kind, instructions, criteria = self._question("question", request.get("question"), where="question")
        if kind != "noul":
            raise RequestError("per ora si calibrano solo le noul", "question.type")
        examples = request.get("examples")
        if not isinstance(examples, list) or not MIN_CALIBRATION_EXAMPLES <= len(examples) <= MAX_CALIBRATION_EXAMPLES:
            raise RequestError(f"servono da {MIN_CALIBRATION_EXAMPLES} a {MAX_CALIBRATION_EXAMPLES} esempi"
                               " (consigliati 32-64)", "examples")
        z, y = [], []
        for i, ex in enumerate(examples):
            if not isinstance(ex, dict) or "state" not in ex or not isinstance(ex.get("answer"), (bool, int)) \
                    or ex["answer"] not in (0, 1):
                raise RequestError('ogni esempio deve essere {"state": ..., "answer": true/false}', f"examples.{i}")
            if len(as_text(ex["state"])) > MAX_TEXT_CHARS:
                raise RequestError(f"state supera {MAX_TEXT_CHARS} caratteri", f"examples.{i}.state")
            text, labels, _, _, key = self.render("question", kind, instructions, criteria, as_text(ex["state"]))
            z.append(float(self.model.noul_logit(text, labels[0])))
            y.append(int(ex["answer"]))
        if len(set(y)) < 2:
            raise RequestError("servono esempi sia con risposta sì sia con risposta no", "examples")
        cal = self.model.cal
        a, b = fit_platt_map(z, y, (cal["platt_a"], cal["platt_b"]), cal.get("fewshot_sigma", FEWSHOT_SIGMA))
        entry = {"a": a, "b": b, "examples": len(y), "positives": sum(y), "model": self.name,
                 "label": labels[0], "instructions": instructions if criteria is not None else None}
        with self._lock:
            if key not in self._calibrations and len(self._calibrations) >= MAX_CALIBRATIONS:
                raise RequestError(f"raggiunto il massimo di {MAX_CALIBRATIONS} calibrazioni salvate", "question")
            self._calibrations[key] = entry
            self._save()
        return {"id": self._id(key), **entry}

    def calibrations(self):
        with self._lock:
            items = list(self._calibrations.items())
        return [{"id": self._id(k), **v} for k, v in items if v["model"] == self.name]

    @staticmethod
    def _id(key):
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def _save(self):
        if not self.calibrations_path:
            return
        tmp = f"{self.calibrations_path}.tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 1, "calibrations": self._calibrations}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.calibrations_path)
