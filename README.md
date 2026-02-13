# OCR Clarity AI

I started this project as a blunt challenge: take a messy, real-world PDF (multi-page, scanned, skewed, faint ink, mixed handwriting and printed fields) and produce text that is actually usable. 
The constraints werw simple: No paid server. No GPU requirement. One command. 
Then reality hit: insurance paperwork is less “documents”, but more hostile ecosystem of forms, grids, stamps, photocopies of photocopies, and handwriting that looks like it was written during a turbulence landing ^^.

This repo is the result: a locally runnable OCR pipeline that adapts per page, emits structured outputs for databases, and includes production-grade reliability primitives (metrics, deviation checks, queue mode, API mode, hard timeouts).

---

## What this repo contains

Core entrypoints:

- `clarity_ocr.py` : CLI OCR engine. Produces `.txt`, `.jsonl`, `.metrics.json`, `.deviation.json`.
- `clarity_ocr_hard.py` : Hard per-page timeout wrapper. Kills pages that freeze and continues.
- `api_service.py` : FastAPI service to submit jobs and stream progress (SSE).
- `clarity_worker.py` : Redis-backed worker for queue processing with Prometheus metrics.
- `enqueue_job.py` : Submit a job to Redis queue.
- `load_test.py` : Simple load testing utility.
- `engines.py` : Optional extra OCR engines (PaddleOCR, docTR) used automatically when installed.

Files for deployment:

- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
- `requirements-extras.txt` (optional heavy engines)

---

## Installation (local)

### 1) Python and virtual environment

Recommended: Python 3.10 to 3.12 (Windows and Linux).

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 2) Install base requirements

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### 3) Optional: install extra engines (printed/forms accuracy)

This is optional. The pipeline works without it. When installed, the engine router auto-uses them for `FORM` and `PRINTED` pages.

```bash
pip install -r requirements-extras.txt
```

Notes:
- PaddleOCR may require a compatible `paddlepaddle` wheel for your OS/CPU/GPU. If your platform is tricky, start with docTR first.

---

## Installation (Docker)

### Base image (lightweight)

```bash
docker build -t clarity-ocr:base .
```

### Full image (installs extras)

```bash
docker build --build-arg INSTALL_EXTRAS=1 -t clarity-ocr:full .
```

---

## Quickstart (CLI)

### Basic run

```bash
python clarity_ocr.py ./samples/sample_handwritten_medical_notes_5p.pdf --out ./output/out_claim.txt
```

Output files will be created next to `--out`:

- `out_claim.txt`
- `out_claim.jsonl`
- `out_claim.metrics.json`
- `out_claim.deviation.json`
- `clarity_baseline.metrics.json` (created once per output folder)

### Hard per-page timeout (recommended for insurance packs)

Hard timeouts are enforceable: if a page runs longer than the budget, it is killed.

Windows (cmd):

```bat
set OCR_PAGE_TIMEOUT_SECONDS=60
set OCR_HARD_PAGE_TIMEOUT=1
python clarity_ocr_hard.py ./samples/sample_handwritten_medical_notes_5p.pdf --out ./output/out_claim.txt
```

Linux/Mac:

```bash
export OCR_PAGE_TIMEOUT_SECONDS=60
export OCR_HARD_PAGE_TIMEOUT=1
python clarity_ocr_hard.py ./samples/sample_handwritten_medical_notes_5p.pdf --out ./output/out_claim.txt
```

---

## Queue mode (Redis worker)

### Start stack

```bash
docker compose up --build
```

Scale workers:

```bash
docker compose up --scale worker=2
```

Submit a job:

```bash
python enqueue_job.py --pdf ./samples/sample_handwritten_medical_notes_5p.pdf
```

Workers write results to the configured output directory (see env vars below).

---

## API mode (FastAPI + SSE for frontend)

Start the API:

```bash
python -m uvicorn api_service:app --host 127.0.0.1 --port 8000 --log-level debug
```

Typical flow for a React frontend:
- POST `/submit` with PDF
- Subscribe to `/events/{job_id}` using SSE to show progress
- Download results from `/download/{job_id}?kind=txt|jsonl|metrics|deviation`

The API uses the same output contract as the CLI.

---

## How it works (technical pipeline)

For each page, the system tries to be smart before being expensive:

1) **Embedded text extraction**
   - If the PDF page contains extractable text (digital PDF), we capture it directly (fast, high accuracy).

2) **Rasterization**
   - If needed, render page to image using PyMuPDF at a chosen DPI.

3) **Page classification**
   - Determine page mode: `EMBEDDED`, `SCAN`, `PRINTED`, `FORM`, `HANDWRITING` (names may vary by version).
   - Use simple measurable signals: ink ratio, connected components count, grid ratio.

4) **Preprocessing**
   - Grid and line suppression (forms)
   - Thresholding and morphology
   - Connected component filtering (ignore tiny junk)

5) **Engine routing**
   - Handwriting and handwriting-like crops: TrOCR handwritten model.
   - Printed/forms pages: prefer PaddleOCR if installed, else docTR, else fallback to printed TrOCR model.
   - The router is automatic; you can override via env var.

6) **Block emission**
   - Every recognized block emits a JSONL record with page index, bbox, script, engine, and text.
   - The `.txt` output is a readable concatenation.

---

## Engine router (auto-selection)

By default:

- For `FORM` or `PRINTED` pages:
  - PaddleOCR (if available)
  - else docTR (if available)
  - else TrOCR printed
- For handwriting-heavy pages: TrOCR handwritten

Control with:

```bash
export OCR_ENGINE_PREF=auto      # default
export OCR_ENGINE_PREF=paddleocr # force
export OCR_ENGINE_PREF=doctr     # force
export OCR_ENGINE_PREF=trocr     # force fallback
```

---

## Models used

Default models (Transformers):

- Printed: `microsoft/trocr-small-printed`
- Handwriting: `microsoft/trocr-small-handwritten`

The warnings you may see:
- “unauthenticated requests to HF Hub” means downloads are slower. You can set a token to improve rate limits.
- “MISSING pooler weights” appears on some checkpoints and is not fatal for inference.

Production recommendation:
- Pin exact model revisions in your deployment process.
- Treat model upgrades like releases (baseline and compare metrics).

---

## Metrics and deviation (drift detection)

### Metrics file: `*.metrics.json`
Example fields you should expect (high-level):
- pages processed
- pages with errors
- placeholder rate (junk/noise proxy)
- block counts and kind distribution
- script distribution (latin/cyrillic/arabic/cjk/unknown)
- timing stats (mean, p95)

### Baseline
On the first run in an output folder, the system writes:
- `clarity_baseline.metrics.json`

### Deviation file: `*.deviation.json`
Compares current metrics vs baseline and produces alerts such as:
- P1 page errors
- P1 junk spike (placeholder rate jump)
- P2 script mix shift (JS divergence)
- P2 mode shift

How to operate this in production:
- Put metrics in Prometheus (worker already exposes a port).
- Trigger alerts when deviation crosses thresholds.
- Store baselines per document class (claim packs vs invoices vs medical notes).

---

## Output schema (database-ready)

### Text output: `*.txt`
Human readable concatenation of blocks.

### JSONL output: `*.jsonl`
Each line is a JSON object.

Common record types you will see:
- `run_header`
- `block`
- `page_summary`
- `page_error`

A typical `block` example:

```json
{
  "type": "block",
  "run_id": "run_...",
  "doc_id": "doc_...",
  "page": 1,
  "kind": "PRINTED",
  "text": "Policy Number: 12345",
  "script": "latin",
  "bbox": [120, 340, 880, 392],
  "placeholder": false,
  "engine": "paddleocr"
}
```

Database ingestion recommendation:
- Store `run_id`, `doc_id`, `page`, `bbox`, `kind`, `script`, `engine`, `text`.
- Index by `(doc_id, page)` and use bbox for highlighting in viewers.
- Keep `metrics.json` as a run-level table.

---

## Imports used (core technical dependencies)

Inside the core scripts you will typically see imports like:

- PDF and images:
  - `import fitz` (PyMuPDF)
  - `from PIL import Image`
- Image processing:
  - `import cv2`
  - `import numpy as np`
- Models:
  - `import torch`
  - `from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor`
- Progress and concurrency:
  - `from tqdm import tqdm`
  - `from concurrent.futures import ThreadPoolExecutor, as_completed`
- Structured output and utilities:
  - `import json, time, hashlib, os, argparse`
- Queue/API mode:
  - `import redis`
  - `from fastapi import FastAPI, UploadFile`
  - `from prometheus_client import Counter, Histogram`

Optional engines (in `engines.py`):
- PaddleOCR:
  - `from paddleocr import PaddleOCR`
- docTR:
  - `from doctr.models import ocr_predictor`

---

## Environment variables (tuning)

Common knobs:

- `CLARITY_DEVICE` : `auto|cpu|cuda`
- `OCR_ENGINE_PREF` : `auto|paddleocr|doctr|trocr`
- `OCR_WORKERS` : number of parallel page workers (CPU bound on decode)
- `OCR_BATCH_SIZE` : crop batch size for model inference
- `OCR_DPI` : render DPI for scanned PDFs
- `OCR_MAX_CROPS` : crop budget per page
- `OCR_PAGE_TIMEOUT_SECONDS` : used by hard wrapper
- `OCR_HARD_PAGE_TIMEOUT` : `1` to enable hard wrapper auto-switch in worker

Queue/API related:
- `REDIS_URL`
- `CLARITY_QUEUE`, `CLARITY_DLQ`
- `CLARITY_PROM_PORT`

---

## Troubleshooting

### `AttributeError: module 'fitz' has no attribute 'open'`
This usually means you installed the wrong `fitz` pip package.

Fix:

```bash
pip uninstall -y fitz
pip install -U pymupdf
```

### HuggingFace rate limit warning
Set a token (optional):

Windows:
```bat
set HF_TOKEN=your_token_here
```

Linux/Mac:
```bash
export HF_TOKEN=your_token_here
```

### Slow runtime on low-end CPU
TrOCR decoding is expensive. Printed forms can explode into many crops if segmentation is too eager.
Use:
- Optional printed engine (PaddleOCR/docTR)
- Hard timeouts
- Lower DPI
- Lower OCR_MAX_CROPS
- Increase OCR_WORKERS if you have more cores

---

## Roadmap (accuracy without sacrificing speed)

To beat typical OCR repos in insurance reality,  we chase system behavior instead of chasing one model:

- Better form detection, more aggressive grid suppression
- Engine routing improvements using measurable signals
- Confidence scoring and language model cleanup only when needed
- Training loop with labeled claim packs and regression tests
- Drift monitoring tied to document classes and source scanners
- Offline evaluation harness that replays historical PDFs and compares metrics and text deltas

---

## License

MIT
