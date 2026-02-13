#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clarity Hard-Timeout Wrapper
----------------------------
Purpose
- Enforce TRUE (hard) per-page timeouts by running each page OCR in a subprocess.
- Preserves the exact same output contract as the core CLI:
  - .txt, .jsonl, .metrics.json, .deviation.json
- Avoids loading OCR models in the parent process (parent stays lightweight).

Enable
- Set OCR_HARD_PAGE_TIMEOUT=1 and OCR_PAGE_TIMEOUT_SECONDS>0
- The worker and the API stack will automatically use this wrapper when enabled.

Usage
- Same as core CLI:
    python clarity_ocr_hard.py input.pdf --out out.txt

How it works
- Parent creates a run_id and spawns the core CLI for each page:
    python clarity_ocr.py input.pdf --single-page N --skip-run-header --skip-deviation --parent-run-id RUN
- Parent merges child outputs into final .txt/.jsonl
- Parent computes metrics/deviation by scanning merged JSONL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import fitz  # PyMuPDF
from tqdm import tqdm


def now_iso() -> str:
    # compact UTC ISO
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_run_id() -> str:
    return f"run_{int(time.time())}_{os.getpid()}"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def js_divergence(p: Dict[str, int], q: Dict[str, int]) -> float:
    keys = set(p) | set(q)
    ps = sum(p.values()); qs = sum(q.values())
    if ps <= 0 or qs <= 0:
        return 0.0
    def norm(d, s):
        return {k: d.get(k, 0)/s for k in keys}
    P = norm(p, ps); Q = norm(q, qs)
    M = {k: 0.5*(P[k]+Q[k]) for k in keys}
    def kl(A, B):
        s = 0.0
        for k in keys:
            a = A[k]; b = B[k]
            if a > 0 and b > 0:
                s += a * math.log(a/b, 2)
        return s
    return 0.5*kl(P, M) + 0.5*kl(Q, M)


def scan_metrics_from_jsonl(jsonl_path: Path) -> Dict[str, Any]:
    mode_counts: Dict[str, int] = {}
    kind_counts: Dict[str, int] = {}
    script_counts: Dict[str, int] = {}
    placeholders = 0
    blocks = 0
    pages = 0
    pages_with_errors = 0
    latencies = []

    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("type")
            if t == "page_summary":
                pages += 1
                if isinstance(obj.get("latency_ms"), (int, float)):
                    latencies.append(float(obj["latency_ms"]))
                if obj.get("mode"):
                    mode_counts[str(obj["mode"])] = mode_counts.get(str(obj["mode"]), 0) + 1
            elif t == "page_error":
                pages_with_errors += 1
            elif t == "block":
                blocks += 1
                if obj.get("placeholder"):
                    placeholders += 1
                k = obj.get("kind")
                if k:
                    kind_counts[str(k)] = kind_counts.get(str(k), 0) + 1
                sc = obj.get("script")
                if sc:
                    script_counts[str(sc)] = script_counts.get(str(sc), 0) + 1

    error_rate = pages_with_errors / pages if pages else 0.0
    placeholder_rate = placeholders / blocks if blocks else 0.0
    lat_mean = (sum(latencies)/len(latencies)) if latencies else None
    lat_p95 = None
    if latencies:
        xs = sorted(latencies)
        pos = int(math.floor(0.95*(len(xs)-1)))
        lat_p95 = xs[pos]

    return {
        "pages": pages,
        "pages_with_errors": pages_with_errors,
        "error_rate": error_rate,
        "blocks": blocks,
        "placeholders": placeholders,
        "placeholder_rate": placeholder_rate,
        "mode_counts": mode_counts,
        "kind_counts": kind_counts,
        "script_counts": script_counts,
        "timings_ms": {"page_latency_mean": lat_mean, "page_latency_p95": lat_p95},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Input PDF")
    ap.add_argument("--out", default=None, help="Output TXT path (defaults to input stem next to PDF)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"[clarity] input not found: {pdf_path}", file=sys.stderr)
        return 2

    out_txt = Path(args.out).resolve() if args.out else pdf_path.with_suffix(".txt")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_txt.with_suffix(".jsonl")
    metrics_path = out_txt.with_suffix(".metrics.json")
    deviation_path = out_txt.with_suffix(".deviation.json")
    baseline_path = out_txt.parent / "clarity_baseline.metrics.json"

    timeout_s = int(os.environ.get("OCR_PAGE_TIMEOUT_SECONDS", "0"))
    if timeout_s <= 0:
        print("[clarity] hard wrapper requires OCR_PAGE_TIMEOUT_SECONDS>0", file=sys.stderr)
        return 2

    run_id = make_run_id()
    doc_hash = sha256_file(pdf_path)
    doc_id = f"doc_{doc_hash[:16]}"

    core_cli = Path(__file__).with_name("clarity_ocr.py").resolve()

    # Open PDF for page count only
    d = fitz.open(str(pdf_path))
    total_pages = d.page_count
    d.close()

    tmpdir = Path(tempfile.mkdtemp(prefix="clarity_hard_"))
    try:
        # Create temp outputs
        txt_tmp = out_txt.with_suffix(out_txt.suffix + ".tmp")
        jsonl_tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")

        with open(txt_tmp, "w", encoding="utf-8") as tf, open(jsonl_tmp, "w", encoding="utf-8") as jf:
            # Run header (single)
            run_header = {
                "ts_utc": now_iso(),
                "type": "run_header",
                "run_id": run_id,
                "doc_id": doc_id,
                "source_pdf": str(pdf_path),
                "doc_sha256": doc_hash,
                "pages": total_pages,
                "device": os.environ.get("CLARITY_DEVICE", "auto"),
                "hard_timeout": True,
                "page_timeout_s": timeout_s,
            }
            jf.write(json.dumps(run_header, ensure_ascii=False) + "\n")

            bar = tqdm(total=total_pages, desc="Pages (hard)", leave=True)
            fatal = 0

            for page_no in range(1, total_pages+1):
                child_txt = tmpdir / f"page_{page_no:04d}.txt"
                # child writes jsonl next to txt
                child_jsonl = child_txt.with_suffix(".jsonl")

                env = os.environ.copy()
                env["OCR_HARD_PAGE_TIMEOUT"] = "0"  # prevent recursion

                cmd = [
                    sys.executable, str(core_cli),
                    str(pdf_path),
                    "--out", str(child_txt),
                    "--single-page", str(page_no),
                    "--skip-run-header",
                    "--skip-deviation",
                    "--parent-run-id", run_id,
                ]

                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
                    rc = int(proc.returncode)

                    # Merge txt
                    if child_txt.exists():
                        tf.write(child_txt.read_text(encoding="utf-8", errors="ignore"))
                        tf.write("\n")
                    else:
                        tf.write(f"=== PAGE {page_no} / {total_pages} (error) ===\n[error] child produced no txt output\n\n")
                        fatal += 1

                    # Merge jsonl (no run_header expected)
                    if child_jsonl.exists():
                        for line in child_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
                            if line.strip():
                                jf.write(line + "\n")
                    else:
                        jf.write(json.dumps({
                            "ts_utc": now_iso(),
                            "type": "page_error",
                            "run_id": run_id,
                            "doc_id": doc_id,
                            "page": page_no,
                            "phase": "hard_timeout_merge",
                            "error": "child produced no jsonl output",
                            "diagnostics": {},
                        }, ensure_ascii=False) + "\n")
                        fatal += 1

                    if rc not in (0, 10):
                        fatal += 1

                    bar.set_postfix({"page": page_no, "rc": rc, "ms": int((time.perf_counter()-t0)*1000)})
                    bar.update(1)

                except subprocess.TimeoutExpired:
                    jf.write(json.dumps({
                        "ts_utc": now_iso(),
                        "type": "page_error",
                        "run_id": run_id,
                        "doc_id": doc_id,
                        "page": page_no,
                        "phase": "hard_timeout",
                        "error": f"hard page timeout after {timeout_s}s",
                        "diagnostics": {"timeout_s": timeout_s},
                    }, ensure_ascii=False) + "\n")
                    tf.write(f"=== PAGE {page_no} / {total_pages} (timeout) ===\n[error] hard page timeout after {timeout_s}s\n\n")
                    fatal += 1
                    bar.set_postfix({"page": page_no, "rc": "timeout"})
                    bar.update(1)

            bar.close()

        # Atomic rename
        txt_tmp.replace(out_txt)
        jsonl_tmp.replace(out_jsonl)

        # Metrics from merged JSONL
        m = scan_metrics_from_jsonl(out_jsonl)
        metrics = {
            "ts_utc": now_iso(),
            "clarity_version": "hard-wrapper",
            "run_id": run_id,
            "doc_id": doc_id,
            "source_pdf": str(pdf_path),
            "doc_sha256": doc_hash,
            **m
        }
        write_json(metrics_path, metrics)

        # Baseline + deviation
        baseline_created = False
        baseline = None
        if baseline_path.exists():
            try:
                baseline = json.loads(baseline_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                baseline = None
        else:
            baseline = metrics
            baseline_created = True
            write_json(baseline_path, baseline)

        deviation = {
            "ts_utc": now_iso(),
            "run_id": run_id,
            "doc_id": doc_id,
            "baseline_path": str(baseline_path),
            "baseline_created": baseline_created,
            "alerts": [],
            "deltas": {},
        }
        if baseline and not baseline_created:
            # simple deltas + JS shifts
            deviation["deltas"]["placeholder_rate_abs"] = float(metrics.get("placeholder_rate", 0.0)) - float(baseline.get("placeholder_rate", 0.0))
            deviation["deltas"]["error_rate_abs"] = float(metrics.get("error_rate", 0.0)) - float(baseline.get("error_rate", 0.0))
            deviation["deltas"]["js_script"] = js_divergence(baseline.get("script_counts", {}), metrics.get("script_counts", {}))
            deviation["deltas"]["js_mode"] = js_divergence(baseline.get("mode_counts", {}), metrics.get("mode_counts", {}))

            if metrics.get("pages_with_errors", 0) > 0:
                deviation["alerts"].append({"id": "P1_PAGE_ERRORS", "why": f"{metrics['pages_with_errors']} pages failed"})
            if baseline.get("placeholder_rate") is not None and metrics.get("placeholder_rate", 0.0) > max(0.08, float(baseline["placeholder_rate"]) * 1.8):
                deviation["alerts"].append({"id": "P1_JUNK_SPIKE", "why": "placeholder spike vs baseline"})
            if deviation["deltas"]["js_script"] > 0.18:
                deviation["alerts"].append({"id": "P2_SCRIPT_MIX_SHIFT", "why": f"JS={deviation['deltas']['js_script']:.3f}"})
            if deviation["deltas"]["js_mode"] > 0.18:
                deviation["alerts"].append({"id": "P2_PAGE_MODE_SHIFT", "why": f"JS={deviation['deltas']['js_mode']:.3f}"})

        write_json(deviation_path, deviation)

        print(f"[clarity] out: {out_txt}")
        print(f"[clarity] jsonl: {out_jsonl}")
        print(f"[clarity] metrics: {metrics_path}")
        print(f"[clarity] deviation: {deviation_path}")
        print(f"[clarity] baseline: {baseline_path} (created={baseline_created})")

        # Exit code: 0 ok, 10 partial, 2 fail
        if metrics.get("pages_with_errors", 0) > 0:
            return 10
        return 0

    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
