#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Push a job to the Clarity Redis queue.

Usage:
  python enqueue_job.py --pdf ./samples/in/claim.pdf --out ./samples/out --job-id claim-001
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import redis  # type: ignore

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.environ.get("CLARITY_QUEUE", "clarity:jobs")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--out", required=None)
    p.add_argument("--job-id", required=None)
    p.add_argument("--out-name", default=None)
    return p.parse_args()

def main() -> int:
    a = parse_args()
    pdf = Path(a.pdf).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "job_id": a.job_id,
        "pdf_path": str(pdf),
        "out_dir": str(out),
        "out_name": a.out_name,
        "attempt": 0,
    }
    r = redis.from_url(REDIS_URL, decode_responses=False)
    r.lpush(QUEUE_NAME, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    print(f"[enqueued] {payload}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
