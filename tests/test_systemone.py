"""Test dell'endpoint System One: contratto, resa fissata, formule di confidenza, validazione.

I test unitari usano un modello finto (token = parole) e non hanno bisogno di un bundle. Il test di integrazione
gira solo se SKEJJA_TEST_BUNDLE punta a un bundle ONNX estratto (es. releases/skejja-edge32k-ft6_onnx).
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from skejja.systemone import ELLIPSIS, RequestError, SystemOne, choice_confidence, score_confidence


class FakeTok:
    def encode(self, text):
        return SimpleNamespace(ids=text.split())

    def decode(self, ids):
        return " ".join(ids)


class FakeModel:
    """Probabilità proporzionali alle parole in comune fra testo ed etichetta; registra le chiamate."""

    def __init__(self):
        self.tok = FakeTok()
        self.calls = []
        self.cal = {"platt_a": 1.0, "platt_b": 0.0}

    def _p(self, text, labels):
        z = np.array([len(set(text.lower().split()) & set(l.lower().split())) for l in labels], np.float64)
        p = np.exp(z - z.max())
        return p / p.sum()

    def choice(self, text, labels):
        self.calls.append(("choice", text, labels))
        return self._p(text, labels)

    def noul_logit(self, text, label):
        return 2.0 if set(label.lower().split()) & set(text.lower().split()) else -2.0

    def noul(self, text, label):
        self.calls.append(("noul", text, [label]))
        z = self.cal["platt_a"] * self.noul_logit(text, label) + self.cal["platt_b"]
        return 1.0 / (1.0 + np.exp(-z))

    def score(self, text, labels, values):
        self.calls.append(("score", text, labels))
        p = self._p(text, labels)
        return float((p * np.asarray(values, np.float64)).sum()), p


@pytest.fixture
def engine():
    return SystemOne(FakeModel(), name="skejja-test")


def test_choice_contract_and_rendering(engine):
    out = engine.run({"model": "jev-latest", "state": "please turn on the lights",
                      "questions": {"azione": {"type": "choice", "instructions": "What does the user want?",
                                               "criteria": {"lights_on": "turn on the lights",
                                                            "music": "play music",
                                                            "lights_off": None}}}})
    assert out["model"] == "skejja-test"
    a = out["answers"]["azione"]
    assert a["type"] == "choice" and a["choice"] == "lights_on"
    assert set(a["probabilities"]) == {"lights_on", "music", "lights_off"}
    assert sum(a["probabilities"].values()) == pytest.approx(1.0)
    assert a["confidence"] == pytest.approx(choice_confidence(list(a["probabilities"].values())))
    _, text, labels = engine.model.calls[0]
    # istruzioni prima dello stato; descrizioni come etichette nell'ordine dato; il nome solo se manca la descrizione
    assert text == "What does the user want?\n\nplease turn on the lights"
    assert labels == ["turn on the lights", "play music", "lights off"]
    assert out["usage"]["output_tokens"] == 0 and out["usage"]["input_tokens"] > 0
    assert out["skejja"] == {"max_tokens": 512, "truncated": [], "calibration": {}}


def test_noul_with_and_without_criteria(engine):
    out = engine.run({"state": "I want a human agent now",
                      "questions": {
                          "con": {"type": "noul", "instructions": "Escalation?",
                                  "criteria": {"true": "asks for a human agent", "false": "anything else"}},
                          "senza": {"type": "noul", "instructions": "Does the user want a refund?"}}})
    assert out["answers"]["con"] == {"type": "noul", "noul": pytest.approx(1 / (1 + np.exp(-2)))}
    assert set(out["answers"]["senza"]) == {"type", "noul"}
    assert out["skejja"]["calibration"] == {"con": "global", "senza": "global"}
    (_, t1, l1), (_, t2, l2) = engine.model.calls
    assert l1 == ["asks for a human agent"] and t1.startswith("Escalation?")
    # senza criteri la domanda è l'etichetta e nel testo resta solo lo stato
    assert l2 == ["Does the user want a refund?"] and t2 == "I want a human agent now"


def test_score_contract(engine):
    levels = ["not urgent", {"what": "somewhat urgent"}, "very urgent now"]
    out = engine.run({"state": "this is very urgent", "questions": {"u": {"type": "score", "criteria": levels}}})
    a = out["answers"]["u"]
    assert a["type"] == "score"
    assert set(a["probabilities"]) == {"0", "1", "2"}
    p = [a["probabilities"][str(i)] for i in range(3)]
    assert a["score"] == pytest.approx(sum(i * v for i, v in enumerate(p)))
    assert a["legend"] == {"0": "not urgent", "1": {"what": "somewhat urgent"}, "2": "very urgent now"}
    assert a["confidence"] == pytest.approx(score_confidence(p))
    assert engine.model.calls[0][2][1] == '{"what": "somewhat urgent"}'


def test_confidence_formulas_match_jev_docs():
    # esempio della pagina Score di Jev: probabilità 0, 0,57, 0,43 → confidenza 0,35
    assert score_confidence([0.0, 0.57, 0.43]) == pytest.approx(0.355, abs=1e-3)
    assert score_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert score_confidence([1 / 3] * 3) == 0.0
    assert choice_confidence([0.5, 0.25, 0.25]) == pytest.approx(0.25)
    assert choice_confidence([0.5, 0.5]) == pytest.approx(0.0)


def test_json_state_and_truncation():
    eng = SystemOne(FakeModel(), max_tokens=64)
    long_state = {"log": " ".join(f"w{i}" for i in range(500))}
    out = eng.run({"state": long_state,
                   "questions": {"q": {"type": "choice", "criteria": {"a": "alpha", "b": "beta"}}}})
    assert out["skejja"]["truncated"] == ["q"] and out["skejja"]["max_tokens"] == 64
    assert out["usage"]["input_tokens"] <= 64
    _, text, _ = eng.model.calls[0]
    assert text.startswith('{"log": "w0') and ELLIPSIS in text


@pytest.mark.parametrize("request_, param", [
    ({"questions": {"q": {"type": "noul", "instructions": "x?"}}}, "state"),
    ({"state": "s", "questions": {}}, "questions"),
    ({"state": "s", "questions": {"q": {"type": "rank"}}}, "questions.q.type"),
    ({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "solo una"}}}}, "questions.q.criteria"),
    ({"state": "s", "questions": {"q": {"type": "score", "criteria": [str(i) for i in range(11)]}}},
     "questions.q.criteria"),
    ({"state": "s", "questions": {"q": {"type": "noul"}}}, "questions.q.instructions"),
    ({"state": "s", "questions": {"q": {"type": "noul", "criteria": {"false": "no"}}}}, "questions.q.criteria"),
    ({"state": "x" * 200_001, "questions": {"q": {"type": "noul", "instructions": "x?"}}}, "state"),
])
def test_validation(engine, request_, param):
    with pytest.raises(RequestError) as e:
        engine.run(request_)
    assert e.value.param == param


QUESTION = {"type": "noul", "instructions": "Does the user want a refund?"}
EXAMPLES = ([{"state": f"I want a refund for order {i}", "answer": True} for i in range(6)]
            + [{"state": f"where is my parcel number {i}", "answer": False} for i in range(6)])


def test_question_calibration_is_used_and_reported(engine):
    before = engine.run({"state": "a refund please", "questions": {"r": QUESTION}})
    cal = engine.calibrate({"question": QUESTION, "examples": EXAMPLES})
    assert cal["examples"] == 12 and cal["positives"] == 6 and cal["model"] == "skejja-test" and len(cal["id"]) == 12
    after = engine.run({"state": "a refund please",
                        "questions": {"r": QUESTION, "altra": {"type": "noul", "instructions": "Is it urgent?"}}})
    assert after["skejja"]["calibration"] == {"r": "question", "altra": "global"}
    z = 2.0   # logit del modello finto quando etichetta e testo hanno parole in comune
    assert after["answers"]["r"]["noul"] == pytest.approx(1 / (1 + np.exp(-(cal["a"] * z + cal["b"]))))
    assert after["answers"]["r"]["noul"] != pytest.approx(before["answers"]["r"]["noul"])
    assert engine.calibrations() == [cal]


def test_calibration_saved_and_scoped_to_model(tmp_path):
    path = tmp_path / "cal.json"
    SystemOne(FakeModel(), name="m1", calibrations_path=str(path)).calibrate({"question": QUESTION,
                                                                             "examples": EXAMPLES})
    req = {"state": "refund", "questions": {"r": QUESTION}}
    assert SystemOne(FakeModel(), name="m1", calibrations_path=str(path)).run(req)["skejja"]["calibration"] == \
        {"r": "question"}
    # un altro .onnx ha un'altra scala dei logit: la calibrazione non si applica
    assert SystemOne(FakeModel(), name="m2", calibrations_path=str(path)).run(req)["skejja"]["calibration"] == \
        {"r": "global"}


@pytest.mark.parametrize("request_, param", [
    ({"question": QUESTION, "examples": EXAMPLES[:4]}, "examples"),
    ({"question": QUESTION, "examples": EXAMPLES[:6] * 2}, "examples"),
    ({"question": {"type": "choice", "criteria": {"a": "x", "b": "y"}}, "examples": EXAMPLES}, "question.type"),
    ({"question": QUESTION, "examples": EXAMPLES[:11] + [{"state": "x", "answer": "forse"}]}, "examples.11"),
    ({"question": QUESTION, "examples": EXAMPLES * 22}, "examples"),
])
def test_calibration_validation(engine, request_, param):
    with pytest.raises(RequestError) as e:
        engine.calibrate(request_)
    assert e.value.param == param


@pytest.mark.skipif(not os.environ.get("SKEJJA_TEST_BUNDLE"), reason="SKEJJA_TEST_BUNDLE non impostata")
def test_real_bundle():
    from skejja import Skejja

    eng = SystemOne(Skejja(os.environ["SKEJJA_TEST_BUNDLE"], threads=2), name="skejja-real")
    out = eng.run({"state": "accendi la luce del soggiorno",
                   "questions": {
                       "azione": {"type": "choice", "criteria": {"luce_on": "turn on the lights",
                                                                 "luce_off": "turn off the lights",
                                                                 "musica": "play music"}},
                       "luce": {"type": "noul", "instructions": "Does the user want to turn on the lights?"},
                       "musica": {"type": "noul", "instructions": "Does the user want to play music?"},
                       "urgenza": {"type": "score", "criteria": ["not urgent", "somewhat urgent", "very urgent"]}}})
    a = out["answers"]
    assert a["azione"]["choice"] == "luce_on"
    assert a["luce"]["noul"] > 0.5 > a["musica"]["noul"]
    assert 0.0 <= a["urgenza"]["score"] <= 2.0
    json.dumps(out)   # serializzabile così com'è


@pytest.mark.skipif(not os.environ.get("SKEJJA_TEST_BUNDLE"), reason="SKEJJA_TEST_BUNDLE non impostata")
def test_real_bundle_truncation_stays_in_budget():
    from skejja import Skejja

    eng = SystemOne(Skejja(os.environ["SKEJJA_TEST_BUNDLE"], threads=2), max_tokens=128)
    state = " ".join(["Il cliente scrive di nuovo per il rimborso, l'ordine 4471 non è mai arrivato."] * 40)
    out = eng.run({"state": state, "questions": {"q": {"type": "choice", "criteria": {
        "rimborso": "asks for a refund", "spedizione": "asks about shipping", "altro": "something else"}}}})
    assert out["skejja"]["truncated"] == ["q"]
    assert out["usage"]["input_tokens"] <= 128


@pytest.mark.skipif(not os.environ.get("SKEJJA_TEST_BUNDLE"), reason="SKEJJA_TEST_BUNDLE non impostata")
def test_real_bundle_calibration():
    from skejja import Skejja

    eng = SystemOne(Skejja(os.environ["SKEJJA_TEST_BUNDLE"], threads=2))
    q = {"type": "noul", "instructions": "Does the user want to turn on the lights?"}
    pos = ["accendi la luce in cucina", "turn on the lights please", "enciende la luz", "mach das Licht an",
           "allume la lumière du salon", "luce accesa in camera, grazie", "lights on in the bedroom",
           "puoi accendere le luci del bagno?"]
    neg = ["spegni la luce", "che tempo fa domani", "play some jazz", "apaga la luz", "wie spät ist es",
           "mets de la musique", "imposta una sveglia alle sette", "turn off the lights"]
    cal = eng.calibrate({"question": q, "examples": [{"state": t, "answer": True} for t in pos]
                         + [{"state": t, "answer": False} for t in neg]})
    assert cal["examples"] == 16
    out = eng.run({"state": "accendi la luce del corridoio", "questions": {"q": q}})
    off = eng.run({"state": "spegni la luce del corridoio", "questions": {"q": q}})
    assert out["skejja"]["calibration"] == {"q": "question"}
    assert out["answers"]["q"]["noul"] > 0.5 > off["answers"]["q"]["noul"]


def test_calibration_store_is_capped(monkeypatch):
    import skejja.systemone as so

    monkeypatch.setattr(so, "MAX_CALIBRATIONS", 2)
    eng = SystemOne(FakeModel())
    for i in range(2):
        eng.calibrate({"question": {"type": "noul", "instructions": f"Question {i} refund?"}, "examples": EXAMPLES})
    eng.calibrate({"question": {"type": "noul", "instructions": "Question 0 refund?"}, "examples": EXAMPLES})  # già presente
    with pytest.raises(RequestError):
        eng.calibrate({"question": {"type": "noul", "instructions": "Question 2 refund?"}, "examples": EXAMPLES})
