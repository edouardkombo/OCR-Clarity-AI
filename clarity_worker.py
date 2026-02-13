#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clarity Worker (Redis queue + Prometheus metrics)

Purpose
- Run Clarity OCR at scale without guessing volume.
- Uses Redis lists as a durable-ish queue (BRPOP).
- Writes results to a results hash and supports a dead-letter queue.
- Exposes Prometheus metrics on :8000 by default.

Job contract (JSON pushed into Redis list)
{
  "job_id": "claim-1234",
  "pdf_path": "/data/in/claim.pdf",
  "out_dir": "/data/out",
  "out_name": "claim_1234"  # optional, default derived from filename
}

Notes
- Worker executes the CLI as a subprocess for isolation and reliability.
- CLI still works standalone: python clarity_ocr.py input.pdf
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import redis  # type: ignore
from prometheus_client import Counter, Histogram, Gauge, start_http_server  # type: ignore


# -----------------------------
# Metrics
# -----------------------------

JOBS_TOTAL = Counter("clarity_jobs_total", "Total jobs consumed", ["status"])
JOB_SECONDS = Histogram("clarity_job_seconds", "Job runtime seconds", buckets=(1,2,3,5,8,13,21,34,55,89,144,233,377,610))
OCR_EXIT_CODE = Counter("clarity_ocr_exit_code_total", "OCR exit codes", ["code"])
INFLIGHT = Gauge("clarity_inflight", "Jobs currently being processed")
QUEUE_LAG = Gauge("clarity_queue_lag", "Approx queue length")

# -----------------------------
# Config
# -----------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = os.environ.get("CLARITY_QUEUE", "clarity:jobs")
DLQ_NAME = os.environ.get("CLARITY_DLQ", "clarity:dlq")
RESULTS_HASH = os.environ.get("CLARITY_RESULTS_HASH", "clarity:results")
EVENTS_STREAM = os.environ.get("CLARITY_EVENTS_LIST", "clarity:events")
EVENTS_PREFIX = os.environ.get("CLARITY_EVENTS_PREFIX", "clarity:events:")

PROM_PORT = int(os.environ.get("CLARITY_PROM_PORT", "8000"))

# Execution
PYTHON = os.environ.get("CLARITY_PYTHON", sys.executable)
CORE_CLI_PATH = str(Path(__file__).with_name("clarity_ocr.py"))
HARD_CLI_PATH = str(Path(__file__).with_name("clarity_ocr_hard.py"))
CLI_PATH = os.environ.get("CLARITY_CLI_PATH", HARD_CLI_PATH if os.environ.get("OCR_HARD_PAGE_TIMEOUT","0").strip() in {"1","true","yes"} else CORE_CLI_PATH)
MAX_RETRIES = int(os.environ.get("CLARITY_MAX_RETRIES", "2"))

# Timeouts
JOB_TIMEOUT_SECONDS = int(os.environ.get("CLARITY_JOB_TIMEOUT_SECONDS", "3600"))  # 1 hour ceiling by default


@dataclass
class Job:
    job_id: str
    pdf_path: str
    out_dir: str
    out_name: Optional[str] = None
    attempt: int = 0


def parse_job(raw: bytes) -> Job:
    obj = json.loads(raw.decode("utf-8", errors="ignore"))
    return Job(
        job_id=str(obj.get("job_id") or ""),
        pdf_path=str(obj.get("pdf_path") or ""),
        out_dir=str(obj.get("out_dir") or ""),
        out_name=(str(obj.get("out_name")) if obj.get("out_name") else None),
        attempt=int(obj.get("attempt") or 0),
    )


def build_cmd(job: Job) -> Tuple[list[str], Path, Path]:
    pdf = Path(job.pdf_path).resolve()
    out_dir = Path(job.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = job.out_name or pdf.stem
    out_txt = out_dir / f"{out_name}.txt"
    cmd = [PYTHON, CLI_PATH, str(pdf), "--out", str(out_txt)]
    return cmd, out_txt, out_txt.with_suffix(".jsonl")


def _job_events_key(job_id: str) -> str:
    return f"{EVENTS_PREFIX}{job_id}"


def push_event(r: redis.Redis, typ: str, payload: Dict[str, Any]) -> None:
    evt = {"ts": time.time(), "type": typ, **payload}
    r.lpush(EVENTS_STREAM, json.dumps(evt, ensure_ascii=False))
    jid = payload.get("job_id")
    if jid:
        r.lpush(_job_events_key(str(jid)), json.dumps(evt, ensure_ascii=False))


def run_job(r: redis.Redis, job: Job) -> None:
    t0 = time.perf_counter()
    INFLIGHT.set(1)
    status = "ok"
    code = 0

    try:
        cmd, out_txt, out_jsonl = build_cmd(job)
        push_event(r, "job_started", {"job_id": job.job_id, "pdf_path": job.pdf_path, "out_txt": str(out_txt), "attempt": job.attempt})
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT_SECONDS)
        code = int(proc.returncode)
        OCR_EXIT_CODE.labels(code=str(code)).inc()

        # Interpret code: 0 success, 10 partial, else fail
        if code == 0:
            status = "success"
        elif code == 10:
            status = "partial"
        else:
            status = "failed"

        result = {
            "job_id": job.job_id,
            "status": status,
            "exit_code": code,
            "pdf_path": job.pdf_path,
            "out_txt": str(out_txt),
            "out_jsonl": str(out_jsonl),
            "stdout": (proc.stdout[-4000:] if proc.stdout else ""),
            "stderr": (proc.stderr[-4000:] if proc.stderr else ""),
            "attempt": job.attempt,
            "runtime_s": round(time.perf_counter() - t0, 3),
        }
        r.hset(RESULTS_HASH, job.job_id, json.dumps(result, ensure_ascii=False))
        push_event(r, "job_finished", result)
        JOBS_TOTAL.labels(status=status).inc()

        # Retry logic
        if status == "failed" and job.attempt < MAX_RETRIES:
            job.attempt += 1
            push_event(r, "job_retry", {"job_id": job.job_id, "attempt": job.attempt})
            r.lpush(QUEUE_NAME, json.dumps(job.__dict__, ensure_ascii=False))
        elif status == "failed":
            r.lpush(DLQ_NAME, json.dumps(job.__dict__, ensure_ascii=False))

    except subprocess.TimeoutExpired:
        status = "timeout"
        JOBS_TOTAL.labels(status=status).inc()
        push_event(r, "job_timeout", {"job_id": job.job_id, "timeout_s": JOB_TIMEOUT_SECONDS, "attempt": job.attempt})
        r.lpush(DLQ_NAME, json.dumps(job.__dict__, ensure_ascii=False))
    except Exception as e:
        status = "crash"
        JOBS_TOTAL.labels(status=status).inc()
        push_event(r, "job_crash", {"job_id": job.job_id, "error": f"{type(e).__name__}: {e}", "attempt": job.attempt})
        r.lpush(DLQ_NAME, json.dumps(job.__dict__, ensure_ascii=False))
    finally:
        INFLIGHT.set(0)
        JOB_SECONDS.observe(max(0.0, time.perf_counter() - t0))


def main() -> int:
    r = redis.from_url(REDIS_URL, decode_responses=False)
    start_http_server(PROM_PORT)
    print(f"[worker] redis={REDIS_URL} queue={QUEUE_NAME} dlq={DLQ_NAME} prom=:{PROM_PORT}")
    print(f"[worker] cli={CLI_PATH} python={PYTHON} max_retries={MAX_RETRIES} timeout_s={JOB_TIMEOUT_SECONDS}")

    stop = {"flag": False}

    def _sig(_signo, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    while not stop["flag"]:
        try:
            # update queue length gauge (approx)
            try:
                QUEUE_LAG.set(float(r.llen(QUEUE_NAME)))
            except Exception:
                pass

            item = r.brpop(QUEUE_NAME, timeout=2)
            if not item:
                continue
            _, raw = item
            job = parse_job(raw)
            if not job.job_id or not job.pdf_path or not job.out_dir:
                # malformed job, send to DLQ
                r.lpush(DLQ_NAME, raw)
                continue

            existing = r.hget(RESULTS_HASH, job.job_id)
            if existing:
                try:
                    prev = json.loads(existing)
                    if prev.get('status') in {'success','partial'}:
                        push_event(r, 'job_skipped_idempotent', prev)
                        JOBS_TOTAL.labels(status='idempotent_skip').inc()
                        continue
                except Exception:
                    pass

            run_job(r, job)

        except Exception as e:
            # keep the loop alive
            print(f"[worker] loop error: {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(1)

    print("[worker] stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
