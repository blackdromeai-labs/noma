"""Noma server: `/v1/systemone` (TypeSafe/Jev wire format) plus a local playground at `/`.

    noma serve                                   # BlackdromeAILabs/noma from Hugging Face
    noma serve --model /path/to/exported/noma    # a local export
    NOMA_CKPT=runs/noma uvicorn noma.serve.app:app # a training checkpoint (development)

Accepts the native shape (fields at the top level) and the wrapped shape ({"input": {...}}).
`probabilities` always sums to 1 over the caller's options; abstention and ensemble
uncertainty are reported separately under `noma`.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

MODEL_NAME = os.environ.get("NOMA_MODEL_NAME", "noma")
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(_app):
    if os.environ.get("NOMA_LAZY") != "1":  # load in the background so the playground opens at once
        threading.Thread(target=lambda: _safe(model), daemon=True).start()
    yield


app = FastAPI(title="Noma", version="1.0.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

_model = None
_state = {"ready": False, "error": None, "device": None, "fast_path": False}
_lock = threading.Lock()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load():
    global _model
    from noma.model.noma import Noma
    dev = os.environ.get("NOMA_DEVICE") or pick_device()
    ckpt = os.environ.get("NOMA_CKPT")
    if ckpt:
        m = Noma.load(ckpt).to(dev).eval()
    else:
        m = Noma.from_pretrained(os.environ.get("NOMA_MODEL", "BlackdromeAILabs/noma"), device=dev)
    # Fast path (length buckets + CUDA graphs) needs an NVIDIA GPU; elsewhere Noma runs the
    # plain path, which is exact but slower.
    bucket = int(os.environ.get("NOMA_BUCKET", "128" if dev == "cuda" else "0"))
    if bucket and dev == "cuda":
        m.bucket = bucket
        m.warmup(graphs=os.environ.get("NOMA_GRAPHS", "1") == "1")
        _state["fast_path"] = True
    _model = m
    _state.update(ready=True, device=torch.cuda.get_device_name(0) if dev == "cuda" else dev)


def model():
    with _lock:
        if _model is None and _state["error"] is None:
            try:
                _load()
            except Exception as e:  # reported by /v1/info and the playground
                _state["error"] = f"{type(e).__name__}: {e}"
    if _model is None:
        raise HTTPException(503, f"model not available: {_state['error'] or 'loading'}")
    return _model


def _safe(fn):
    try:
        fn()
    except Exception:
        pass


def _validate(questions) -> dict:
    if not isinstance(questions, dict) or not questions:
        raise HTTPException(400, "'questions' must be a non-empty object")
    for key, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "noul", "score"):
            raise HTTPException(400, f"question {key!r}: type must be choice, noul, or score")
        crit = q.get("criteria")
        if q["type"] == "choice" and not (isinstance(crit, dict) and 1 <= len(crit) <= 255):
            raise HTTPException(400, f"question {key!r}: choice criteria must be an object of 1-255 options")
        if q["type"] == "score" and not (isinstance(crit, list) and len(crit) >= 2):
            raise HTTPException(400, f"question {key!r}: score criteria must be a list of >= 2 levels")
        if q["type"] == "noul" and crit is not None and not isinstance(crit, dict):
            raise HTTPException(400, f"question {key!r}: noul criteria must be an object")
    return questions


def answer(q: dict, probs: dict, abstain: float, unc: float) -> dict:
    ext = {"abstain": round(abstain, 6), "uncertainty": round(unc, 6)}
    if q["type"] == "noul":
        p = probs["true"]
        return {"type": "noul", "noul": p, "confidence": max(p, 1 - p),
                "probabilities": probs, "noma": ext}
    top = max(probs, key=probs.get)
    if q["type"] == "score":
        return {"type": "score", "score": sum(int(k) * v for k, v in probs.items()),
                "confidence": probs[top], "probabilities": probs, "noma": ext}
    return {"type": "choice", "choice": top, "confidence": probs[top], "probabilities": probs,
            "noma": ext}


@app.post("/v1/systemone")
async def systemone(req: Request):
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "body must be JSON")
    if isinstance(body, dict) and isinstance(body.get("input"), dict):
        body = body["input"]
    if not isinstance(body, dict) or "state" not in body:
        raise HTTPException(400, "missing 'state'")
    questions = _validate(body.get("questions"))
    m = model()
    ref = body.get("reference_time")
    if not ref:
        ref = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cuda = next(m.parameters()).is_cuda
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        out, n_tokens = m.decide(body["state"], questions, ref)
    if cuda:
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    answers = {k: answer(questions[k], *out[k]) for k in questions}
    resp = {"model": MODEL_NAME, "answers": answers,
            "usage": {"input_tokens": n_tokens, "output_tokens": 0},
            "noma": {"model_ms": round(ms, 2)}}
    if not body.get("reference_time"):
        resp["noma"]["reference_time"] = f"not given; used server clock {ref}"
    return resp


@app.get("/v1/info")
def info():
    return {"model": MODEL_NAME, **_state}


@app.get("/health")
def health():
    return {"ok": _state["ready"], "model": MODEL_NAME}


@app.get("/")
def playground():
    return FileResponse(STATIC / "index.html")
