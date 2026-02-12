# OCR Clarity AI

I started this as a challenge to reach highest accuracy with most optimal extraction time of insurance documents with ideally no GPU.

I discovered that insurance PDFs are a small universe of their own: multi-page bundles, mixed quality, random rotations, photocopy generations deep, and handwriting that looks like it was written during turbulence. I wanted one thing: **PDF in, usable text out**, locally, without needing a server, without needing a GPU, and without me babysitting a zoo of flags.

So I jumped on the challenge.

I began with synthetic messy handwritten pages (because that's the controlled laboratory). It worked. I felt clever for about five minutes.

Then reality arrived, something I haven't thought about: real insurance "documents" are often **scanned forms**. Boxes. Tables. Grid lines. Tiny printed fields. Handwritten fill-ins squeezed into rectangles. My neat line segmentation suddenly exploded into nonsense. 
A page that should have produced 40 lines produced 400 "lines", because the grid lines looked like text. 

This repo is the result: **one-command OCR that adapts automatically**.

---

## What it does

Run one command:

```bash
python ocr_clarity.py input.pdf
```

It writes:

- `input.txt`

And it adapts per page:

1) **If the PDF already contains embedded selectable text**  
   It extracts that text directly. No OCR. Fast. Accurate.

2) **If the page is a scan**  
   It rasterizes the page, cleans it, detects whether it is:
   - mostly handwriting (notes pages)
   - a form/grid-heavy scan (claim forms, medical forms, invoices with tables)

3) **If it is a form**  
   It removes grid lines, segments into regions, and routes regions to:
   - printed OCR model for printed-looking text blocks
   - handwritten OCR model for handwriting-looking text blocks

4) **If it is handwriting**  
   It uses line-based segmentation and OCR optimized for handwriting.

All of this happens automatically. No arguments required beyond the input file.

---

## Why it can be slow right now (even if it is quite fast depending on your machine power)

## Speed upgrades in this version

I wanted speed without denial (highest accuracy). So I added a parallel conveyor belt:

- **Page preparation runs in parallel**: render, preprocess, classify, segment, build crops.
- **Inference stays in one place**: batching stays efficient and GPU stays stable.
- **The pipeline is bounded**: it only prepares a small number of pages ahead, so memory does not balloon on long PDFs.
- **Empty form fields are skipped before OCR**: that prevents classic empty-field hallucinations like `0 000 000 ...`.

By default it uses a small thread pool and a small prefetch window, to tune it without adding more CLI flags:

- `OCR_WORKERS=4` controls how many pages can be prepared in parallel.
- `OCR_PREFETCH=2` controls how many pages are prepared ahead.

Example:

```bash
OCR_WORKERS=4 OCR_PREFETCH=2 python ocr_clarity.py input.pdf
```


The current behavior is accuracy-first, and that has a cost.

On scanned PDFs, the pipeline does a lot of work per page:

### 1) It has to understand the page structure before reading it
Every scanned page goes through:
- denoise + normalization
- adaptive thresholding (ink vs background)
- deskew
- "Is this a form?" detection (grid line estimation + segmentation sanity checks)
- segmentation (either line bands or connected-component regions)
- optional printed-vs-handwritten routing per region (for forms)

This is the price of not hallucinating 400 lines out of a table.

### 2) Transformer OCR is expensive because it decodes token-by-token
TrOCR is an encoder-decoder transformer. For each detected region, it performs autoregressive decoding. That means:
- more regions = more model calls
- more model calls = more time

For multi-page documents, the total time is basically:

**time ≈ pages × regions_per_page × decode_cost**

This is why a "simple looking" 12-page claim bundle can take longer than expected. 
The document is not big because of pages, but only because of "regions".

### 3) The tool is adaptive, so it runs checks to stay out of trouble
It tries to avoid OCR when embedded text exists, and it tries to avoid segmentation strategies that explode on forms. That adaptivity improves accuracy, but adds compute.

---

## What I did not think about at first (and learned fast)

I assumed "handwriting OCR" meant:
- take page
- split into lines
- decode

But insurance paperwork is rarely a page of free handwriting. It is often:
- a form with grid lines
- small printed labels
- handwriting inside boxes
- stamps, signatures, and scan artifacts

So the first surprise was segmentation: **grid lines are a really hostile geometry** ^^.

The second surprise was multi-page behavior: a single PDF can contain:
- page 1: a clean digital letter with embedded text
- page 2: a scanned form
- page 3: a handwritten note
- page 4: an invoice with tables

Treating the whole PDF as one type is how OCR pipelines die quietly.

This is why this repository makes decisions per page.

---

## What it does NOT do yet

### Language detection is not implemented (on purpose)
Right now, the OCR step does not detect language. I left it out for technical reasons:

1) OCR errors distort language detection  
   If the OCR output is noisy, language classifiers still confidently return an answer, so we get false certainty.

2) Language detection does not automatically improve OCR  
   Unless you also route to language/script-specific OCR models, language detection is mostly useful downstream.

3) Real multi-language support means script-aware routing  
   Latin handwriting, Arabic, CJK, Cyrillic, etc often need different OCR strategies. Detecting language without changing OCR behavior is a checkbox, not an improvement.

Planned approach:
- add language detection **after OCR** at the page or document level
- use that to choose downstream NLP models (NER, summarization, validation rules)
- only add OCR script routing when we commit to multi-model OCR across scripts

**Please, feel free to contribute to this repository to augment it**

---

## Why these libraries (pure technical reasons)

### PyMuPDF (`pymupdf`)
- Fast PDF rendering and embedded text extraction
- No external system dependencies like poppler
- Page-by-page processing keeps memory stable for multi-page PDFs

### OpenCV (`opencv-python-headless`)
- Native C++ implementations for morphology, thresholding, connected components
- Crucial for form line removal and segmentation
- Headless build avoids GUI dependencies, better for servers and CI

### Pillow (`pillow`)
- Reliable image representation and easy bridge into model inputs
- Lightweight, widely compatible

### Transformers + SentencePiece (`transformers[sentencepiece]`, `sentencepiece`, `protobuf`)
- Required to load and run Microsoft TrOCR models
- SentencePiece is necessary for the tokenizer used by TrOCR variants

### Torch (`torch`)
- Inference runtime
- Enables GPU acceleration when available
- Allows mixed precision on GPU for speed and memory reduction

### tqdm (`tqdm`)
- Progress bars for long multi-page documents
- Without it, "is it stuck?" becomes a daily ritual

---

## How to install

Recommended: Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\\Scripts\\activate   # Windows

pip install -U pip setuptools wheel
pip install -r requirements.txt
```

GPU note:
- GPU support depends on installing the correct torch build for your CUDA version.
- The script automatically uses GPU if torch reports CUDA available.

---

## How to run

Minimal:

```bash
python ocr_clarity.py input.pdf
```

Optional:
```bash
python ocr_clarity.py input.pdf --out output.txt
python ocr_clarity.py input.pdf --dpi 350
python ocr_clarity.py input.pdf --max-pages 20
```

---

## How I plan to make it faster without losing accuracy

The goal is not to "optimize" by ripping out adaptivity, which I think, will just make it fast at being wrong.

The real and more senseful plan, is to reduce wasted work while keeping the same correctness guardrails.

### Phase 0: Make the analysis cheaper (big gains, minimal risk)
1) Downscale for page classification  
   - Detect grids and page type on a downscaled copy (example: width 1200 px)
   - Keep full DPI only for OCR crops

2) Hard early exits  
   - If a page is strongly form-like, skip handwriting projection segmentation entirely
   - If strongly handwriting-like, skip connected-components segmentation entirely

3) Stop region explosions early  
   - If segmentation produces an extreme number of regions, merge regions into larger chunks
   - It is often better to OCR 80 sensible regions than 800 tiny ones

4) Cache page-level decisions in multi-page PDFs  
   - Many claim bundles repeat layout across pages
   - If consecutive pages look similar (grid ratio + ink density), reuse segmentation strategy and parameters

### Phase 1: Make OCR throughput higher (especially on GPU)
1) Dynamic batch sizing that expands, not only shrinks  
   - Today we back off on out-of-memory
   - Next: also probe upward when there is VRAM headroom

2) Stabilize crop sizes  
   - Resize crops into a small set of canonical shapes
   - More consistent shapes improve batching efficiency and reduce overhead

3) Optional `torch.compile()` on supported environments  
   - Useful when inference pattern is repetitive and stable
   - Not mandatory, but a clean speed lever for power users

4) Optional quantization for CPU mode  
   - Quantization can cut CPU inference time significantly
   - Needs careful validation to avoid accuracy regressions on handwriting

### Phase 2: Pipeline parallelism (for long PDFs)
Multi-page documents naturally lend themselves to a pipeline:
- Stage A: render pages
- Stage B: preprocess + segment
- Stage C: OCR inference
- Stage D: write output

A producer-consumer pipeline lets A/B overlap with C, improving throughput without changing OCR correctness.

### Phase 3: Layout-aware extraction for true insurance-grade performance
Heuristics are good, but layout models can reduce region count dramatically:
- detect text blocks, tables, key-value zones
- OCR fewer, better regions
- preserve reading order and structure more reliably

This is I think the path to both speed and accuracy at scale. Any additional suggestions are welcome

---

## How I decided to think about accuracy improvements (practical, high ROI)

If accuracy is low, most engineers will target the model, but experience told me, it is usually not the model first, but the input physics:

1) Scan quality:
- 300 to 350 DPI
- avoid overexposure (faint ink disappears)
- avoid blur (handwriting becomes entropy)

2) Straightness:
- deskew matters
- rotated text destroys segmentation

3) Forms:
- grid line removal matters
- otherwise segmentation mistakes multiply downstream errors

This repository is built around those realities.

---


## Where the model comes from (and how to keep it honest in production)

This project does not hide a secret in-house model. The "brain" is two **pretrained TrOCR checkpoints** loaded through Hugging Face `transformers`:

- **Printed text**: `microsoft/trocr-small-printed`
- **Handwriting**: `microsoft/trocr-small-handwritten`

On first run, `transformers` downloads the weights into the local Hugging Face cache (then reuses them). 
In other words: **the repo can run entirely on a laptop**, but the *first* run needs internet to fetch the model files unless you pre-bundle them.

### Production rule 1: pin the exact model revision

Do not only reference a model by name please, othwerise it means you are implicitly trusting "whatever the Hub serves today", and this should never be done!
Why? It is fine for experiments, but in production it makes results non-reproducible.

The fix is simple:
- pin a **specific revision hash** for each model,
- log it for every run,
- treat that pinned snapshot as immutable.

That gives us a more scientific baseline: same inputs, same code, same model weights, same output. 
If anything changes, we can then prove *what* changed.

### Production rule 2: treat OCR as a pipeline tunable without retraining

Most OCR regressions in insurance do not come from the model suddenly forgetting English as I said, but from **input shift**:

- new scanner settings (contrast, sharpness, compression)
- shadowing, blur, skew, rotation
- different form templates (new grid thickness, new checkboxes)
- different handwriting styles (pen type, pressure, slant)
- multi-generation photocopies

That is why `ocr_clarity.py` is opinionated: it routes pages into modes (embedded / printed / handwriting / form) and does segmentation before OCR. 
In production, we can recover a lot of accuracy by tuning the pipeline first.

**Tuning ladder (cheapest to most powerful):**
1) **Pipeline tuning (no training)**: thresholds, segmentation kernels, line removal, routing heuristics, "skip empty fields" ink ratio. Fastest path to a real win.
2) **Model swap (no training)**: move from `trocr-small-*` to `trocr-base-*` for accuracy (slower, heavier), or swap printed OCR engine for more pure throughput.
3) **Fine-tuning (training)**: collect insurance-specific ground truth and adapt TrOCR to your exact documents.

### How to detect drift in accuracy (two kinds of drift)

I'll define Drift as two different monsters wearing the same coat.

#### A) Quality drift (accuracy actually dropped)

We need a small, continuously evaluated "gold set" (representative, versioned):

- 10–20 printed pages
- 10–20 handwriting pages
- 10–20 form pages with mixed printed labels + handwritten fields
- multilingual examples if your business needs it

Then compute metrics regularly:
- **CER** (Character Error Rate) for handwriting
- **WER** (Word Error Rate) for printed text
- **Field-level accuracy** for business-critical fields: policy number, claim id, DOB, invoice number, totals

If these metrics cross a threshold, we have real quality drift.

#### B) Input drift (the world changed, labels not available yet)

Even without human-labeled ground truth, we can detect "something is different" by monitoring distributions:

- DPI / page size distribution
- blur score (ex: Laplacian variance)
- skew angle estimates
- ink density ratio per page and per crop
- number of detected regions per page
- routing ratio: printed vs handwriting vs form
- percent of regions skipped as empty
- count of placeholder-like outputs blocked (should be near-zero in healthy docs)

When these stats move sharply, the input distribution changed, and quality drift is likely to follow.

### Updating the model safely (canary, receipts, rollback)

When drift is detected, I think the response should be boringly disciplined.

1) **Triage**
- Is it printed only? handwriting only? forms only?
- Is it one partner's scanners? one specific template?

2) **Quick patch first**
- adjust preprocessing/segmentation thresholds
- strengthen line removal for the new form template
- tighten/loosen the empty-field skip rules
- update routing heuristics

3) **Collect hard examples**
- sample failures
- get human transcription for the exact failure cases
- keep them as a new "hard set"

4) **Fine-tune (optional, when the pipeline can't save anybody)**
- train a new snapshot `insurance-htr-v2`
- evaluate on the gold set + hard set
- compare against the old snapshot

5) **Canary deploy**
- run old vs new in parallel on a small traffic slice
- promote only if metrics improve (and no new failure modes appear)

6) **Rollback is mandatory**
- keep the previous model snapshot available
- if your drift monitor alarms after deployment, roll back immediately

### What I log per OCR run (so debugging is physics, not vibes)

To run this in production, log these with every output in order to audit later:

- model ids (printed + handwriting) + pinned revision hashes
- code version (git commit)
- preprocessing config hash (thresholds, kernels, routing rules)
- device info (cpu/cuda/mps), dtype, batch size
- per-page mode decisions (embedded/printed/handwriting/form)
- region counts and skip counts

This way, if at one point, "it got worse last week", we can answer with data rather than assumptions.


## Outputs and next steps

Current output:
- plain text per page

Next obvious upgrade:
- structured JSON output with:
  - page index
  - region coordinates
  - extracted text
  - routing type (printed vs handwriting)
  - confidence signals

That turns raw OCR into something an insurance workflow can trust, audit, and improve.

---

## License
MIT

---

## TL;DR
I started with handwriting because it was the clean experiment.
I hit scanned forms and learned what insurance documents actually are.
Now it adapts page-by-page, runs locally, uses GPU automatically when available,
and it is honest about the current speed tax: segmentation + region-level decoding.

The next iteration is about speed without denial.

---

## Troubleshooting (the two classic booby traps)

### `AttributeError: module 'fitz' has no attribute 'open'`
This almost always means you installed the wrong package.

- ✅ Correct: `pymupdf` (this provides the `fitz` module internally)
- ❌ Wrong: the separate PyPI package named `fitz`

Fix:
```bash
pip uninstall -y fitz
pip install -U pymupdf
```

### Tokenizer / model cache weirdness (SentencePiece)
TrOCR relies on SentencePiece. If you see tokenizer errors, update these:
```bash
pip install -U sentencepiece "transformers[sentencepiece]"
```



## Autopilot speed (automatic, no extra flags)

If you see minutes per page, the bottleneck is usually not "a page". It is **the number of OCR crops**. A single scanned form can explode into hundreds of tiny regions, and each region costs decoder time.

Clarity OCR AI adds an autopilot tuner:

- **Low DPI first** for fast classification and segmentation (`OCR_LOW_DPI`, default 200)
- **High DPI only when handwriting is present** (`OCR_HIGH_DPI`, default max(--dpi, 300))
- **Region explosion control**: if segmentation produces too many crops, it retries with stronger merging (`OCR_MAX_CROPS`, default 180)
- **Dynamic token budgets**: printed uses fewer tokens than handwriting to reduce decode time

Run it the same way:

```bash
python ocr_clarity.py input.pdf
```

Override autopilot (optional):

```bash
OCR_LOW_DPI=200 OCR_HIGH_DPI=300 OCR_MAX_CROPS=180 python ocr_clarity.py input.pdf
```


## Output sanity (why you might see "000 000 000" and how Clarity suppresses it)

In messy insurance scans, segmentation sometimes extracts **near-empty boxes** (blank form fields, underlines, grid fragments, checkboxes).
TrOCR is a generative decoder, so on low-information crops it can produce pathological patterns like:

- `1 000 000 000 000 ...`
- long symbol soup
- repeated short phrases (looping)

Clarity OCR AI addresses this in two layers:

1) **Stricter "ink gating" before OCR** (skip near-empty crops)
   - handwriting crops default to higher ink thresholds
   - form handwriting crops default to higher ink thresholds
   - printed/form crops use mode-specific thresholds

2) **A stronger placeholder filter before writing output**
   - detects long `000` group chains (even if they start with 1 or 2)
   - detects low character diversity on long strings
   - detects alphabet soup sequences (a b c d ...)
   - detects word-level repetition loops

If your handwriting is extremely faint, you can relax thresholds:

```bash
OCR_HW_MIN_RATIO=0.0018 OCR_HW_MIN_PIXELS=110 python ocr_clarity.py input.pdf
```
