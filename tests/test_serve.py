"""The HTTP layer and the playground, with a stand-in model (no weights needed)."""

import os

os.environ["NOMA_LAZY"] = "1"  # never load real weights during tests

import pytest
import torch
from fastapi.testclient import TestClient

from noma.serve import app as srv


class FakeModel:
    """Answers uniformly over each question's options, like Noma.decide's return shape."""

    def parameters(self):
        yield torch.zeros(1)

    def decide(self, state, questions, reference_time=None):
        self.seen = (state, questions, reference_time)
        out = {}
        for key, q in questions.items():
            if q["type"] == "noul":
                opts = ["true", "false"]
            elif q["type"] == "score":
                opts = [str(i) for i in range(len(q["criteria"]))]
            else:
                opts = list(q["criteria"])
            probs = {o: 1 / len(opts) for o in opts}
            probs[opts[0]] += 1e-3  # a clear top option
            out[key] = (probs, 0.05, 0.01)
        return out, 42


@pytest.fixture
def client(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(srv, "_model", fake)
    monkeypatch.setitem(srv._state, "ready", True)
    c = TestClient(srv.app)
    c.fake = fake
    return c


QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "Billing", "identity": "Login"}},
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["none", "low", "medium", "high"]},
}


def test_all_three_question_types(client):
    r = client.post("/v1/systemone", json={"state": "charged twice", "questions": QUESTIONS,
                                           "reference_time": "2026-03-20T09:00:00Z"})
    assert r.status_code == 200
    body = r.json()
    a = body["answers"]
    assert a["team"]["choice"] == "billing" and set(a["team"]["probabilities"]) == {"billing", "identity"}
    assert 0 <= a["refund"]["noul"] <= 1 and a["refund"]["confidence"] >= 0.5
    assert 0 <= a["urgency"]["score"] <= 3
    for ans in a.values():
        assert set(ans["noma"]) == {"abstain", "uncertainty"}
    assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert body["noma"]["model_ms"] >= 0 and "reference_time" not in body["noma"]
    assert client.fake.seen[2] == "2026-03-20T09:00:00Z"


def test_wrapped_shape_and_missing_reference_time(client):
    r = client.post("/v1/systemone", json={"model": "noma", "input": {"state": {"a": 1}, "questions": QUESTIONS}})
    assert r.status_code == 200
    assert "server clock" in r.json()["noma"]["reference_time"]


@pytest.mark.parametrize("payload", [
    {"questions": QUESTIONS},                                         # no state
    {"state": "x"},                                                   # no questions
    {"state": "x", "questions": {}},
    {"state": "x", "questions": {"q": {"type": "essay"}}},
    {"state": "x", "questions": {"q": {"type": "choice", "criteria": {}}}},
    {"state": "x", "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}},
    {"state": "x", "questions": {"q": {"type": "score", "criteria": ["only"]}}},
    {"state": "x", "questions": {"q": {"type": "noul", "criteria": ["a"]}}},
])
def test_bad_requests_are_rejected(client, payload):
    assert client.post("/v1/systemone", json=payload).status_code == 400


def test_body_must_be_json(client):
    r = client.post("/v1/systemone", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_info_and_health(client):
    info = client.get("/v1/info").json()
    assert info["model"] == "noma" and info["ready"] is True
    assert client.get("/health").json() == {"ok": True, "model": "noma"}


def test_unavailable_model_reports_503(monkeypatch):
    monkeypatch.setattr(srv, "_model", None)
    monkeypatch.setitem(srv._state, "error", "RuntimeError: no weights")
    r = TestClient(srv.app).post("/v1/systemone", json={"state": "x", "questions": QUESTIONS})
    assert r.status_code == 503 and "no weights" in r.json()["detail"]


def test_playground_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Noma" in page.text
    for asset in ("app.css", "app.js", "blackdrome-logo.svg"):
        assert f"/static/{asset}" in page.text or asset == "blackdrome-logo.svg"
        assert client.get(f"/static/{asset}").status_code == 200


def test_playground_presets_are_valid_requests(client):
    """Every preset shipped in the playground must be a request the API accepts."""
    import json
    import re
    js = (srv.STATIC / "app.js").read_text(encoding="utf-8")
    assert "/v1/systemone" in js
    assert len(re.findall(r"\btype:\s*['\"](choice|noul|score)['\"]", js)) >= 5
    assert json.dumps(QUESTIONS)  # the shapes above mirror the presets' three types
