#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clarity API Service (FastAPI)
-----------------------------
Purpose
- Provide a stable HTTP interface for frontends (React or otherwise) to submit OCR jobs,
  monitor progress, and download outputs.
- Keeps CLI as the source of truth: worker executes the CLI, so outputs match exactly.

Key features
- Idempotency: doc_sha256-based job dedup and result reuse.
- SSE progress stream: frontend can wait and receive events without polling.
- Volume-agnostic: you can scale worker replicas horizontally.

Run (with docker-compose):
  docker compose up --build
Then:
  POST /submit (multipart) or /submit_path (server-side path)
  GET /status/{job_id}
  GET /events/{job_id}  (SSE stream)
  GET /download/{job_id}?kind=txt|jsonl|metrics|deviation
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import redis  # type: ignore
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = os.environ.get("CLARITY_QUEUE", "clarity:jobs")
RESULTS_HASH = os.environ.get("CLARITY_RESULTS_HASH", "clarity:results")

# Per-job events list. Worker pushes here.
EVENTS_PREFIX = os.environ.get("CLARITY_EVENTS_PREFIX", "clarity:events:")

DATA_IN = Path(os.environ.get("CLARITY_DATA_IN", "/data/in")).resolve()
DATA_OUT = Path(os.environ.get("CLARITY_DATA_OUT", "/data/out")).resolve()
DATA_IN.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Clarity OCR API", version="1.0")

r = redis.from_url(REDIS_URL, decode_responses=False)

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def result_for_job(job_id: str) -> Optional[dict]:
    raw = r.hget(RESULTS_HASH, job_id)
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return None

def enqueue(job: dict) -> None:
    r.lpush(QUEUE_NAME, json.dumps(job, ensure_ascii=False).encode("utf-8"))

def event_key(job_id: str) -> str:
    return f"{EVENTS_PREFIX}{job_id}"

def push_event(job_id: str, typ: str, payload: Dict[str, Any]) -> None:
    evt = {"ts": time.time(), "type": typ, "job_id": job_id, **payload}
    r.lpush(event_key(job_id), json.dumps(evt, ensure_ascii=False).encode("utf-8"))

@app.post("/submit")
async def submit(file: UploadFile = File(...)) -> JSONResponse:
    b = await file.read()
    if not b:
        raise HTTPException(400, "Empty upload")

    doc_hash = sha256_bytes(b)
    job_id = f"doc-{doc_hash[:16]}"

    # Idempotency: if we already have results for this doc, return them
    existing = result_for_job(job_id)
    if existing and existing.get("status") in {"success", "partial"}:
        return JSONResponse({"job_id": job_id, "idempotent": True, "result": existing})

    # Save file to DATA_IN
    pdf_path = DATA_IN / f"{job_id}.pdf"
    pdf_path.write_bytes(b)

    out_name = job_id
    job = {"job_id": job_id, "pdf_path": str(pdf_path), "out_dir": str(DATA_OUT), "out_name": out_name, "attempt": 0}
    push_event(job_id, "enqueued", {"pdf_path": str(pdf_path), "out_dir": str(DATA_OUT)})
    enqueue(job)
    return JSONResponse({"job_id": job_id, "idempotent": False, "pdf_path": str(pdf_path), "out_dir": str(DATA_OUT)})

@app.post("/submit_path")
async def submit_path(pdf_path: str) -> JSONResponse:
    p = Path(pdf_path).resolve()
    if not p.exists() or p.suffix.lower() != ".pdf":
        raise HTTPException(400, "pdf_path must exist and end with .pdf")

    b = p.read_bytes()
    doc_hash = sha256_bytes(b)
    job_id = f"doc-{doc_hash[:16]}"

    existing = result_for_job(job_id)
    if existing and existing.get("status") in {"success", "partial"}:
        return JSONResponse({"job_id": job_id, "idempotent": True, "result": existing})

    # copy into DATA_IN for stable container paths
    dst = DATA_IN / f"{job_id}.pdf"
    if dst != p:
        dst.write_bytes(b)

    job = {"job_id": job_id, "pdf_path": str(dst), "out_dir": str(DATA_OUT), "out_name": job_id, "attempt": 0}
    push_event(job_id, "enqueued", {"pdf_path": str(dst), "out_dir": str(DATA_OUT)})
    enqueue(job)
    return JSONResponse({"job_id": job_id, "idempotent": False, "pdf_path": str(dst), "out_dir": str(DATA_OUT)})

@app.get("/status/{job_id}")
async def status(job_id: str) -> JSONResponse:
    res = result_for_job(job_id)
    if res:
        return JSONResponse(res)
    return JSONResponse({"job_id": job_id, "status": "queued_or_running"})

@app.get("/events/{job_id}")
async def events(job_id: str) -> StreamingResponse:
    """
    Server-Sent Events stream.
    Frontend can listen until it sees job_finished (success/partial/failed).
    """
    key = event_key(job_id)

    async def gen():
        # Send initial comment to open stream
        yield b": ok\n\n"
        last_sent = 0
        while True:
            # Prefer BLPOP style waiting using polling for simplicity.
            raw = r.brpop(key, timeout=2)
            if raw:
                _, payload = raw
                yield b"data: " + payload + b"\n\n"
                last_sent = time.time()
            else:
                # keep-alive every ~10s
                if time.time() - last_sent > 10:
                    yield b": keep-alive\n\n"
                    last_sent = time.time()

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/download/{job_id}")
async def download(job_id: str, kind: str) -> FileResponse:
    res = result_for_job(job_id)
    if not res:
        raise HTTPException(404, "No result yet")
    # worker writes these output paths
    out_txt = Path(res.get("out_txt", ""))
    if not out_txt.exists():
        raise HTTPException(404, "Output not found")
    mapping = {
        "txt": out_txt,
        "jsonl": out_txt.with_suffix(".jsonl"),
        "metrics": out_txt.with_suffix(".metrics.json"),
        "deviation": out_txt.with_suffix(".deviation.json"),
    }
    if kind not in mapping:
        raise HTTPException(400, "kind must be txt|jsonl|metrics|deviation")
    p = mapping[kind]
    if not p.exists():
        raise HTTPException(404, f"{kind} not found yet")
    return FileResponse(p, filename=p.name)

