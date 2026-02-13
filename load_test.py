#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple load test:
- Enqueue N jobs pointing to same PDF (different job_ids), wait for completion.
- Prints throughput and success ratio.

Usage:
  python load_test.py --pdf ./samples/in/your.pdf --out ./samples/out --n 20
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import redis  # type: ignore

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.environ.get("CLARITY_QUEUE", "clarity:jobs")
RESULTS_HASH = os.environ.get("CLARITY_RESULTS_HASH", "clarity:results")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=10)
    return p.parse_args()

def main():
    a = parse_args()
    pdf = Path(a.pdf).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    r = redis.from_url(REDIS_URL, decode_responses=False)

    job_ids = []
    t0 = time.time()
    for i in range(a.n):
        jid = f"load-{int(t0)}-{i}"
        job = {"job_id": jid, "pdf_path": str(pdf), "out_dir": str(out), "out_name": jid, "attempt": 0}
        r.lpush(QUEUE_NAME, json.dumps(job, ensure_ascii=False).encode("utf-8"))
        job_ids.append(jid)
    print(f"Enqueued {len(job_ids)} jobs")

    done = set()
    ok = 0
    partial = 0
    failed = 0
    while len(done) < len(job_ids):
        for jid in job_ids:
            if jid in done:
                continue
            raw = r.hget(RESULTS_HASH, jid)
            if not raw:
                continue
            res = json.loads(raw.decode("utf-8", errors="ignore"))
            st = res.get("status")
            done.add(jid)
            if st == "success":
                ok += 1
            elif st == "partial":
                partial += 1
            else:
                failed += 1
        time.sleep(0.5)

    dt = time.time() - t0
    print(f"Completed in {dt:.2f}s | ok={ok} partial={partial} failed={failed} | throughput={len(job_ids)/dt:.2f} jobs/s")

if __name__ == "__main__":
    main()
