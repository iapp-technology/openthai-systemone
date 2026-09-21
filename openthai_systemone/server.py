"""FastAPI server exposing the TypeSafe-compatible endpoint:

    POST /v1/systemone  {"state": ..., "model": "...", "questions": {...}}  ->  {"model", "answers", "usage"}

Run:  OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne uvicorn openthai_systemone.server:app --port 8000
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .client import SystemOneClient
from .types import SystemOneRequest, SystemOneResponse

app = FastAPI(title="OpenThai-SystemOne", version="0.1.0")
app.add_middleware(  # browser playgrounds call the endpoint directly; restrict with OPENTHAI_SYSTEMONE_CORS if needed
    CORSMiddleware, allow_origins=[o for o in os.environ.get("OPENTHAI_SYSTEMONE_CORS", "*").split(",") if o],
    allow_methods=["*"], allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_client() -> SystemOneClient:
    path = os.environ.get("OPENTHAI_SYSTEMONE_MODEL", "iapp/OpenThai-SystemOne")
    return SystemOneClient(path, model_name=os.environ.get("OPENTHAI_SYSTEMONE_NAME", "openthai-systemone"))


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/v1/systemone", response_model=SystemOneResponse)
def system_one(req: SystemOneRequest):
    try:
        n = req.permutations if req.permutations else (8 if req.order_invariant else (1 if req.order_invariant is False else None))
        resp = get_client().system_one(req.state, req.questions, permutations=n)
    except ValidationError as e:  # pragma: no cover
        raise HTTPException(status_code=422, detail=e.errors())
    resp.model = req.model or resp.model
    return resp
