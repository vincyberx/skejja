"""Test del server HTTP: stesso contratto di /v1/systemone, errori 400 strutturati, chiave API facoltativa."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from skejja.server import create_app  # noqa: E402
from skejja.systemone import SystemOne  # noqa: E402
from test_systemone import EXAMPLES, QUESTION, FakeModel  # noqa: E402

REQUEST = {"state": "turn on the lights",
           "questions": {"q": {"type": "choice", "criteria": {"on": "turn on the lights", "off": "music"}}}}


def client(api_key=None):
    return TestClient(create_app(SystemOne(FakeModel(), name="skejja-test"), api_key=api_key))


def test_systemone_ok():
    r = client().post("/v1/systemone", json=REQUEST)
    assert r.status_code == 200
    body = r.json()
    assert body["answers"]["q"]["choice"] == "on" and body["model"] == "skejja-test"


def test_invalid_request_is_400_with_param():
    r = client().post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "rank"}}})
    assert r.status_code == 400
    assert r.json()["error"] == {"type": "invalid_request_error", "message": r.json()["error"]["message"],
                                 "param": "questions.q.type"}


def test_api_key():
    c = client(api_key="segreto")
    assert c.post("/v1/systemone", json=REQUEST).status_code == 401
    assert c.post("/v1/systemone", json=REQUEST, headers={"Authorization": "Bearer sbagliato"}).status_code == 401
    assert c.post("/v1/systemone", json=REQUEST, headers={"Authorization": "Bearer segreto"}).status_code == 200


def test_health():
    assert client().get("/health").json() == {"status": "ok", "model": "skejja-test"}


def test_calibrations_endpoints():
    c = client()
    r = c.post("/v1/skejja/calibrations", json={"question": QUESTION, "examples": EXAMPLES})
    assert r.status_code == 200 and r.json()["examples"] == 12
    assert c.get("/v1/skejja/calibrations").json() == [r.json()]
    body = c.post("/v1/systemone", json={"state": "refund", "questions": {"r": QUESTION}}).json()
    assert body["skejja"]["calibration"] == {"r": "question"}
    assert c.post("/v1/skejja/calibrations", json={"question": QUESTION, "examples": []}).status_code == 400
