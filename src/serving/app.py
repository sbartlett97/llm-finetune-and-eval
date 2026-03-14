from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException

from src.schemas import GenerateRequest, GenerateResponse
from src.serving.model_loader import ModelLoader

logger = logging.getLogger(__name__)

_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _loader.load()
    yield


app = FastAPI(title="LLM Eval Harness", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": _loader.model_run_id}


@app.get("/ready")
def ready() -> dict[str, str]:
    if not _loader.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    if not _loader.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    response_text, tokens_generated = _loader.generate(
        question=request.question,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    return GenerateResponse(
        response=response_text,
        model_run_id=_loader.model_run_id,
        latency_ms=latency_ms,
        tokens_generated=tokens_generated,
    )
