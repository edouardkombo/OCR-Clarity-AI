#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clarity OCR (Production-shaped CLI)
==================================

Purpose
- Extract text from messy multi-page PDFs (insurance-grade scans with printed + handwriting).
- CPU-first, GPU-auto (if CUDA available), bounded memory, streaming outputs.
- Writes both TXT and DB-ready JSONL (one record per block + one per page summary).

Design principles
- Deterministic: no sampling; fixed decoding.
- Reliable: per-page isolation (errors don't kill entire run), atomic output writes, explicit exit codes.
- Observable: progress bars + structured diagnostics per page.

Usage
  python clarity_ocr.py input.pdf
  # optional:
  OCR_WORKERS=2 OCR_PREFETCH=1 python clarity_ocr.py input.pdf
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Optional OCR engines (PaddleOCR/docTR)
try:
    from engines import is_paddle_available, is_doctr_available, ocr_fullpage_paddle, ocr_fullpage_doctr, guess_paddle_lang
except Exception:
    def is_paddle_available() -> bool: return False
    def is_doctr_available() -> bool: return False
    def ocr_fullpage_paddle(*args, **kwargs): return []
    def ocr_fullpage_doctr(*args, **kwargs): return []
    def guess_paddle_lang(script: str) -> str: return "en"
from PIL import Image

# PyMuPDF import: "import fitz" (pymupdf)
import fitz  # type: ignore

import cv2  # type: ignore
import torch  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore

# -----------------------------
# Versioning / constants
# -----------------------------


def jlog(event: str, **fields: Any) -> None:
    """
    Purpose: Emit structured JSON logs for production observability.
    Problem it solves: Text logs are hard to aggregate; JSON logs are machine-friendly.
    Tests to ensure:
      - prints valid JSON lines
      - never throws on serialization issues
    """
    try:
        rec = {"ts_utc": now_iso() if "now_iso" in globals() else datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "event": event, **fields}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    except Exception:
        # last resort fallback
        print(f"[log] {event} {fields}", flush=True)

CLARITY_VERSION = "v8.1.0"
DEFAULT_PRINTED_MODEL = os.environ.get("OCR_MODEL_PRINTED", "microsoft/trocr-small-printed")
DEFAULT_PRINTED_REV = os.environ.get("OCR_MODEL_PRINTED_REV", None)
DEFAULT_HANDWRITTEN_MODEL = os.environ.get("OCR_MODEL_HANDWRITTEN", "microsoft/trocr-small-handwritten")
DEFAULT_HANDWRITTEN_REV = os.environ.get("OCR_MODEL_HANDWRITTEN_REV", None)

# Conservative output junk suppression patterns (expanded from earlier versions)
_PLACEHOLDER_RE_1 = re.compile(r"^(?:0\s*){8,}$")
_PLACEHOLDER_RE_2 = re.compile(r"^[#\-_=\|\.]{8,}$")
_PLACEHOLDER_RE_3 = re.compile(r"^(?:\d\s*){40,}$")

# -----------------------------
# Script detection (Unicode-range)
# -----------------------------

_SCRIPT_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    "greek": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "hebrew": [(0x0590, 0x05FF)],
    "devanagari": [(0x0900, 0x097F)],
    "thai": [(0x0E00, 0x0E7F)],
    "cjk": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x30FF), (0xAC00, 0xD7AF)],
    "latin": [(0x0041, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
}

def detect_primary_script(text: str) -> str:
    """
    Purpose: Tag OCR output with a primary script bucket (latin/cyrillic/arabic/cjk/etc).
    Problem it solves: "all languages" needs indexing + routing without heavyweight language detection.
    Tests to ensure:
      - English/French -> latin
      - Russian -> cyrillic
      - Arabic -> arabic
      - Chinese/Japanese/Korean -> cjk
    """
    if not text:
        return "unknown"
    counts = {k: 0 for k in _SCRIPT_RANGES.keys()}
    for ch in text:
        cp = ord(ch)
        for name, ranges in _SCRIPT_RANGES.items():
            for a, b in ranges:
                if a <= cp <= b:
                    counts[name] += 1
                    break
    best_name, best_count = max(counts.items(), key=lambda kv: kv[1])
    return best_name if best_count > 0 else "unknown"

# -----------------------------
# Small utilities
# -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def safe_doc_id(pdf_path: Path) -> str:
    base = pdf_path.stem.strip().lower()
    base = re.sub(r"[^a-z0-9_\-]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "document"

def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def atomic_open_for_write(path: Path):
    # Write to temp then rename at end for atomicity (best effort on Windows).
    tmp = path.with_suffix(path.suffix + ".tmp")
    f = tmp.open("w", encoding="utf-8", newline="\n")
    return f, tmp

def flush_fsync(f) -> None:
    f.flush()
    try:
        os.fsync(f.fileno())
    except Exception:
        # On some platforms/fs, fsync may fail; flushing is still valuable.
        pass

# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    def to_dict(self) -> dict:
        return {"x0": int(self.x0), "y0": int(self.y0), "x1": int(self.x1), "y1": int(self.y1)}

    def clamp(self, w: int, h: int) -> "Box":
        return Box(
            x0=max(0, min(self.x0, w)),
            y0=max(0, min(self.y0, h)),
            x1=max(0, min(self.x1, w)),
            y1=max(0, min(self.y1, h)),
        )

@dataclass
class CropRecord:
    crop_id: int
    kind: str  # printed|handwriting
    dpi: int
    bbox: Box
    ink_ratio: float
    ink_pixels: int
    is_form: bool
    image: Image.Image

@dataclass
class PagePrepResult:
    page_index: int
    mode: str  # embedded|printed|handwriting|form
    embedded_text: Optional[str]
    crops: List[CropRecord]
    diagnostics: Dict[str, str]
    error: Optional[str] = None

# -----------------------------
# Output sanity filters
# -----------------------------

def is_placeholder_decode(s: str) -> bool:
    """
    Purpose: Prevent polluted lines from appearing in output.
    Problem it solves: OCR on empty/line-only crops can produce junk like '1 000 000 ...' or repetition loops.
    Tests to ensure:
      - drops long 000-chains and symbol soup
      - keeps normal numbers, IDs, and real sentences
    """
    t = (s or "").strip()
    if not t:
        return True

    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return True

    compact = re.sub(r"\s+", "", t)
    if not compact:
        return True

    # pure symbols
    if re.fullmatch(r"[#_\-\=\|\.\,\:\;\/\+\*\(\)\[\]]{3,}", compact):
        return True

    # low diversity on long strings
    if len(compact) >= 40:
        uniq = set(compact.lower())
        if len(uniq) <= 3:
            return True

    # long chains of 000 groups ("1 000 000 000 ...")
    if len(t) >= 30 and re.fullmatch(r"\d(?:\s*000){8,}\s*\d?", t):
        return True

    # dominant zero tokens
    tokens = t.split(" ")
    if len(tokens) >= 10:
        from collections import Counter
        c = Counter(tokens)
        tok, freq = c.most_common(1)[0]
        if freq / float(len(tokens)) >= 0.8 and tok in {"0", "00", "000", "0000"}:
            return True

    # alphabet soup
    if re.fullmatch(r"(?:[A-Za-z]\s+){8,}[A-Za-z]", t):
        return True

    # repetition loops
    words = re.findall(r"[A-Za-z]+", t.lower())
    if len(words) >= 12:
        def max_ngram_repeats(n: int) -> int:
            from collections import Counter
            grams = [" ".join(words[i:i+n]) for i in range(0, len(words) - n + 1)]
            if not grams:
                return 0
            return Counter(grams).most_common(1)[0][1]
        if max_ngram_repeats(2) >= 6 or max_ngram_repeats(3) >= 4:
            return True

    if _PLACEHOLDER_RE_1.fullmatch(t):
        return True
    if _PLACEHOLDER_RE_2.fullmatch(t):
        return True
    if _PLACEHOLDER_RE_3.fullmatch(t):
        return True

    return False


# -----------------------------
# Drift / deviation metrics helpers
# -----------------------------

def _pct(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0

def _quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs2 = sorted(xs)
    if len(xs2) == 1:
        return float(xs2[0])
    pos = (len(xs2) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs2[lo])
    frac = pos - lo
    return float(xs2[lo] * (1 - frac) + xs2[hi] * frac)

def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    def norm(d: Dict[str, float]) -> Dict[str, float]:
        s = sum(d.get(k, 0.0) for k in keys)
        if s <= 0:
            return {k: 0.0 for k in keys}
        return {k: d.get(k, 0.0) / s for k in keys}
    p2 = norm(p); q2 = norm(q)
    m = {k: 0.5 * (p2.get(k, 0.0) + q2.get(k, 0.0)) for k in keys}
    def kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        s = 0.0
        for k in keys:
            x = a.get(k, 0.0)
            y = b.get(k, 0.0)
            if x > 0 and y > 0:
                s += x * math.log(x / y, 2)
        return s
    return 0.5 * kl(p2, m) + 0.5 * kl(q2, m)

def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

# -----------------------------
# Image / PDF handling
# -----------------------------

def pdf_num_pages(pdf_path: Path) -> int:
    with fitz.open(str(pdf_path)) as doc:
        return doc.page_count

def try_extract_embedded_text(page: fitz.Page) -> Optional[str]:
    """
    Purpose: Extract text layer if present.
    Reliability: If extraction fails or is empty, we fall back to OCR.
    """
    try:
        txt = page.get_text("text") or ""
        txt = txt.strip()
        return txt if len(txt) >= 10 else None
    except Exception:
        return None

def page_to_image(page: fitz.Page, dpi: int) -> np.ndarray:
    """
    Purpose: Render a PDF page to a grayscale numpy array.
    Reliability: deterministic rendering; avoids alpha surprises.
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    else:
        gray = img
    return gray

# -----------------------------
# Fast page heuristics: printed vs handwriting vs form
# -----------------------------

def binarize(gray: np.ndarray) -> np.ndarray:
    # Adaptive threshold robust to uneven lighting
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
    return b

def estimate_grid_ratio(bin_img: np.ndarray) -> float:
    # detect lines via morphology; ratio of line pixels to total
    inv = 255 - bin_img
    h, w = inv.shape[:2]
    # horizontal lines
    kx = max(10, w // 60)
    ky = max(10, h // 80)
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)))
    lines = cv2.bitwise_or(horiz, vert)
    return float(np.count_nonzero(lines)) / float(lines.size)

def classify_mode_from_low(bin_low: np.ndarray) -> Tuple[str, Dict[str, str]]:
    """
    Purpose: Classify page mode using cheap low-DPI features.
    Tests: grids -> form, dense strokes -> handwriting, else printed.
    """
    diag: Dict[str, str] = {}
    grid_ratio = estimate_grid_ratio(bin_low)
    diag["grid_ratio"] = f"{grid_ratio:.4f}"

    inv = 255 - bin_low
    ink_ratio = float(np.count_nonzero(inv)) / float(inv.size)
    diag["page_ink_ratio"] = f"{ink_ratio:.4f}"

    # Connected component stats (to detect lots of small strokes)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([], dtype=np.int32)
    diag["cc_count"] = str(int(len(areas)))
    small = int(np.sum((areas >= 10) & (areas <= 120))) if len(areas) else 0
    diag["cc_small"] = str(small)

    # Heuristics
    if grid_ratio >= float(os.environ.get("OCR_GRID_RATIO_FORM", "0.012")):
        return "form", diag
    if ink_ratio >= float(os.environ.get("OCR_PAGE_INK_HW", "0.020")) and small >= int(os.environ.get("OCR_CC_SMALL_HW", "400")):
        return "handwriting", diag
    return "printed", diag

# -----------------------------
# Segmentation and crop control
# -----------------------------

def region_has_enough_ink(bin_crop: np.ndarray, min_ratio: float, min_pixels: int) -> Tuple[bool, float, int]:
    inv = 255 - bin_crop
    ink_pixels = int(np.count_nonzero(inv))
    ink_ratio = ink_pixels / float(inv.size) if inv.size else 0.0
    ok = (ink_ratio >= min_ratio) and (ink_pixels >= min_pixels)
    return ok, float(ink_ratio), int(ink_pixels)

def find_text_regions(bin_img: np.ndarray, mode: str, max_regions: int) -> Tuple[List[Box], List[np.ndarray]]:
    """
    Purpose: Extract candidate text regions as bins + bounding boxes.
    Reliability: bounded by max_regions; includes merge attempt when storms occur.
    """
    inv = 255 - bin_img
    h, w = inv.shape[:2]

    # Morphology parameters based on mode
    if mode == "handwriting":
        # wider merge to join cursive strokes into lines
        k = max(18, w // 55)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3))
    else:
        k = max(12, w // 70)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 2))

    merged = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    boxes: List[Box] = []
    regions: List[np.ndarray] = []
    # sort by y then x (reading order-ish)
    items = []
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area < 50:
            continue
        items.append((y, x, x, y, x + bw, y + bh, area))
    items.sort()

    for _, _, x0, y0, x1, y1, _area in items[: max_regions * 2]:  # soft cap before ink gating
        x0 = max(0, x0 - 3); y0 = max(0, y0 - 3)
        x1 = min(w, x1 + 3); y1 = min(h, y1 + 3)
        crop = bin_img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        boxes.append(Box(x0, y0, x1, y1))
        regions.append(crop)
        if len(boxes) >= max_regions:
            break
    return boxes, regions

def remove_form_lines(bin_img: np.ndarray) -> np.ndarray:
    inv = 255 - bin_img
    h, w = inv.shape[:2]
    kx = max(12, w // 55)
    ky = max(12, h // 70)
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)))
    lines = cv2.bitwise_or(horiz, vert)
    cleaned = inv.copy()
    cleaned[lines > 0] = 0
    out = 255 - cleaned
    return out

def prepare_page(pdf_path: Path, page_index: int, low_dpi: int, high_dpi: int, max_crops: int) -> PagePrepResult:
    """
    Purpose: Prepare a single page: extract embedded text or build OCR crops.
    Reliability: catches exceptions and returns PagePrepResult.error instead of crashing.
    """
    t0 = time.perf_counter()
    try:
        with fitz.open(str(pdf_path)) as doc:
            page = doc.load_page(page_index)
            embedded = try_extract_embedded_text(page)
            if embedded:
                skip_emit = False  # engine router guard

                diag = {"mode": "embedded", "prep_ms": f"{(time.perf_counter()-t0)*1000:.0f}"}
                return PagePrepResult(page_index=page_index, mode="embedded", embedded_text=embedded, crops=[], diagnostics=diag)

            gray_low = page_to_image(page, dpi=low_dpi)
            bin_low = binarize(gray_low)
            mode, diag = classify_mode_from_low(bin_low)

            # form cleanup
            is_form = (mode == "form")
            work = remove_form_lines(bin_low) if is_form else bin_low

            # crop storm caps and retries (merge stronger by increasing max_regions kernel sizes upstream)
            max_regions = max_crops
            boxes_low, regions_low = find_text_regions(work, mode="handwriting" if mode == "handwriting" else "printed", max_regions=max_regions)

            # If crop storms: retry with larger morphology closure (effectively by increasing max_regions?)
            # We'll do a second attempt by re-merging stronger on the bin image before CC.
            if len(boxes_low) >= max_regions:
                diag["storm"] = "1"
            diag["boxes"] = str(len(boxes_low))

            crops: List[CropRecord] = []
            crop_id = 0

            def gate_for(kind: str) -> Tuple[float, int]:
                if kind == "handwriting":
                    return float(os.environ.get("OCR_HW_MIN_RATIO", "0.0022")), int(os.environ.get("OCR_HW_MIN_PIXELS", "140"))
                if is_form:
                    return float(os.environ.get("OCR_FORM_MIN_RATIO", "0.0020")), int(os.environ.get("OCR_FORM_MIN_PIXELS", "180"))
                return float(os.environ.get("OCR_PR_MIN_RATIO", "0.0018")), int(os.environ.get("OCR_PR_MIN_PIXELS", "160"))

            # For handwriting pages, optionally render high DPI once and do line-based crops
            if mode == "handwriting":
                gray_hi = page_to_image(page, dpi=high_dpi)
                bin_hi = binarize(gray_hi)
                # line-ish segmentation using horizontal projections
                inv = 255 - bin_hi
                proj = np.sum(inv > 0, axis=1)
                thresh = max(5, int(0.02 * inv.shape[1]))
                rows = np.where(proj > thresh)[0]
                if rows.size:
                    # group consecutive rows
                    runs = []
                    start = int(rows[0]); prev = int(rows[0])
                    for r in rows[1:]:
                        r = int(r)
                        if r == prev + 1:
                            prev = r
                        else:
                            runs.append((start, prev))
                            start = r; prev = r
                    runs.append((start, prev))
                    # create crops
                    for y0, y1 in runs:
                        y0 = max(0, y0 - 6); y1 = min(bin_hi.shape[0], y1 + 6)
                        reg = bin_hi[y0:y1, :]
                        ok, ir, ip = region_has_enough_ink(reg, *gate_for("handwriting"))
                        if not ok:
                            continue
                        img = Image.fromarray(reg).convert("RGB")
                        crops.append(CropRecord(crop_id=crop_id, kind="handwriting", dpi=high_dpi, bbox=Box(0, y0, bin_hi.shape[1], y1), ink_ratio=ir, ink_pixels=ip, is_form=is_form, image=img))
                        crop_id += 1
                diag["hw_crops"] = str(sum(1 for c in crops if c.kind == "handwriting"))
                diag["pr_crops"] = "0"
                diag["prep_ms"] = f"{(time.perf_counter()-t0)*1000:.0f}"
                return PagePrepResult(page_index=page_index, mode=mode, embedded_text=None, crops=crops, diagnostics=diag)

            # printed or form: use low dpi crops (then later handwriting crops from hi dpi if form)
            for b, reg in zip(boxes_low, regions_low):
                kind = "printed"
                ok, ir, ip = region_has_enough_ink(reg, *gate_for(kind))
                if not ok:
                    continue
                img = Image.fromarray(reg).convert("RGB")
                crops.append(CropRecord(crop_id=crop_id, kind=kind, dpi=low_dpi, bbox=b, ink_ratio=ir, ink_pixels=ip, is_form=is_form, image=img))
                crop_id += 1

            # for forms: also try handwriting-specific extraction at high dpi (filled fields)
            if is_form:
                gray_hi = page_to_image(page, dpi=high_dpi)
                bin_hi = binarize(gray_hi)
                bin_hi = remove_form_lines(bin_hi)
                boxes_hi, regions_hi = find_text_regions(bin_hi, mode="handwriting", max_regions=max(60, max_crops // 2))
                for b, reg in zip(boxes_hi, regions_hi):
                    ok, ir, ip = region_has_enough_ink(reg, float(os.environ.get("OCR_FORM_HW_MIN_RATIO", "0.0022")), int(os.environ.get("OCR_FORM_HW_MIN_PIXELS", "160")))
                    if not ok:
                        continue
                    img = Image.fromarray(reg).convert("RGB")
                    crops.append(CropRecord(crop_id=crop_id, kind="handwriting", dpi=high_dpi, bbox=b, ink_ratio=ir, ink_pixels=ip, is_form=True, image=img))
                    crop_id += 1

            diag["pr_crops"] = str(sum(1 for c in crops if c.kind == "printed"))
            diag["hw_crops"] = str(sum(1 for c in crops if c.kind == "handwriting"))
            diag["prep_ms"] = f"{(time.perf_counter()-t0)*1000:.0f}"
            return PagePrepResult(page_index=page_index, mode=mode, embedded_text=None, crops=crops, diagnostics=diag)

    except Exception as e:
        diag = {"mode": "error", "prep_ms": f"{(time.perf_counter()-t0)*1000:.0f}"}
        return PagePrepResult(page_index=page_index, mode="error", embedded_text=None, crops=[], diagnostics=diag, error=f"{type(e).__name__}: {e}")

# -----------------------------
# Model loading (lazy) + inference
# -----------------------------

class TrocrBundle:
    def __init__(self) -> None:
        self._printed: Optional[Tuple[TrOCRProcessor, VisionEncoderDecoderModel]] = None
        self._hand: Optional[Tuple[TrOCRProcessor, VisionEncoderDecoderModel]] = None

    def printed(self, device: torch.device, dtype: torch.dtype) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
        if self._printed is None:
            proc = TrOCRProcessor.from_pretrained(DEFAULT_PRINTED_MODEL, revision=DEFAULT_PRINTED_REV)
            model = VisionEncoderDecoderModel.from_pretrained(DEFAULT_PRINTED_MODEL, revision=DEFAULT_PRINTED_REV)
            model.to(device=device, dtype=dtype)
            model.eval()
            self._printed = (proc, model)
        return self._printed

    def handwritten(self, device: torch.device, dtype: torch.dtype) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
        if self._hand is None:
            proc = TrOCRProcessor.from_pretrained(DEFAULT_HANDWRITTEN_MODEL, revision=DEFAULT_HANDWRITTEN_REV)
            model = VisionEncoderDecoderModel.from_pretrained(DEFAULT_HANDWRITTEN_MODEL, revision=DEFAULT_HANDWRITTEN_REV)
            model.to(device=device, dtype=dtype)
            model.eval()
            self._hand = (proc, model)
        return self._hand

def choose_device() -> Tuple[torch.device, torch.dtype, str]:
    """
    CPU-first: use GPU if available; choose dtype accordingly.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16, "cuda"
    return torch.device("cpu"), torch.float32, "cpu"

def tokens_for_crop(kind: str, img: Image.Image) -> int:
    w, h = img.size
    area = w * h
    if kind == "handwriting":
        # handwriting often needs longer decoding
        return int(min(196, max(64, area // 2200)))
    return int(min(128, max(48, area // 3200)))

@torch.inference_mode()
def infer_batch(
    images: List[Image.Image],
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    max_new_tokens: int,
    desc: str,
) -> List[str]:
    out: List[str] = []
    if not images:
        return out

    bar = tqdm(total=len(images), desc=desc, leave=False)
    try:
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            px = processor(images=batch, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
            gen = model.generate(
                px,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.12,
            )
            texts = processor.batch_decode(gen, skip_special_tokens=True)
            out.extend([t.strip() for t in texts])
            bar.update(len(batch))
    finally:
        bar.close()
    return out

# -----------------------------
# JSONL writing
# -----------------------------

def write_jsonl_line(jf, record: dict) -> None:
    jf.write(json.dumps(record, ensure_ascii=False) + "\n")

# -----------------------------
# CLI / main
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("pdf", help="Input PDF path")
    p.add_argument("--out", default=None, help="Output TXT path (default: input.txt)")
    # All other settings are optional env vars; we keep CLI simple on purpose.
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"[error] File not found: {pdf_path}", file=sys.stderr)
        return 2
    if pdf_path.suffix.lower() != ".pdf":
        print(f"[error] Input must be a .pdf: {pdf_path}", file=sys.stderr)
        return 2

    out_path = Path(args.out).expanduser().resolve() if args.out else pdf_path.with_suffix(".txt")
    jsonl_path = out_path.with_suffix(".jsonl")

    metrics_path = out_path.with_suffix(".metrics.json")
    deviation_path = out_path.with_suffix(".deviation.json")
    baseline_path = out_path.parent / "clarity_baseline.metrics.json"

    doc_id = safe_doc_id(pdf_path)
    run_id = f"{doc_id}-{int(time.time())}"
    try:
        doc_hash = sha256_file(pdf_path)
    except Exception:
        doc_hash = None

    low_dpi = int(os.environ.get("OCR_LOW_DPI", "200"))
    high_dpi = int(os.environ.get("OCR_HIGH_DPI", "300"))
    max_crops = int(os.environ.get("OCR_MAX_CROPS", "180"))

    # Engine preference: auto -> paddleocr -> doctr -> trocr
    engine_pref = os.environ.get("OCR_ENGINE_PREF", "auto").strip().lower()

    page_timeout_s = int(os.environ.get("OCR_PAGE_TIMEOUT_SECONDS", "0"))  # 0 disables


    # Parallel page prep settings
    workers = int(os.environ.get("OCR_WORKERS", str(max(1, min(4, (os.cpu_count() or 2))))))
    prefetch = int(os.environ.get("OCR_PREFETCH", "2"))
    prefetch = max(1, min(8, prefetch))

    device, dtype, device_tag = choose_device()
    batch_default = int(os.environ.get("OCR_BATCH", "6" if device_tag == "cuda" else "2"))
    batch_default = max(1, min(32, batch_default))

    total_pages = pdf_num_pages(pdf_path)

    print(f"[clarity] {CLARITY_VERSION} | device={device_tag} | pages={total_pages} | workers={workers} | batch={batch_default}")
    print(f"[clarity] models: printed={DEFAULT_PRINTED_MODEL} | handwritten={DEFAULT_HANDWRITTEN_MODEL}")
    print(f"[clarity] out: {out_path}")
    print(f"[clarity] out: {jsonl_path}")

    trocr = TrocrBundle()

    # Open temp outputs for atomic rename at end
    txt_f, txt_tmp = atomic_open_for_write(out_path)
    jf, jsonl_tmp = atomic_open_for_write(jsonl_path)

    # Write run header record
    run_header = {
        "ts_utc": now_iso(),
        "type": "run_header",
        "run_id": run_id,
        "clarity_version": CLARITY_VERSION,
        "doc_id": doc_id,
        "source_pdf": str(pdf_path),
        "doc_sha256": doc_hash,
        "device": device_tag,
        "env": {
            "OCR_LOW_DPI": low_dpi,
            "OCR_HIGH_DPI": high_dpi,
            "OCR_MAX_CROPS": max_crops,
            "OCR_WORKERS": workers,
            "OCR_PREFETCH": prefetch,
            "OCR_BATCH": batch_default,
        },
        "models": {
            "printed": DEFAULT_PRINTED_MODEL,
            "handwritten": DEFAULT_HANDWRITTEN_MODEL,
        },
    }
    if not getattr(args, "skip_run_header", False):
        write_jsonl_line(jf, run_header)
    flush_fsync(jf)

    pages_bar = tqdm(total=total_pages, desc="Pages", leave=True)

    # -----------------------------
    # Aggregates for drift metrics
    # -----------------------------
    agg_page_latency_ms: List[float] = []
    agg_prep_ms: List[float] = []
    agg_total_boxes: List[float] = []
    agg_pr_crops: List[float] = []
    agg_hw_crops: List[float] = []
    agg_ink_ratios: List[float] = []
    agg_block_count = 0
    agg_placeholder_count = 0
    agg_form_blocks = 0
    agg_mode_counts: Dict[str, int] = {}
    agg_kind_counts: Dict[str, int] = {}
    agg_script_counts: Dict[str, int] = {}
    agg_errors: List[dict] = []

    fatal_errors = 0
    try:
        with futures.ThreadPoolExecutor(max_workers=workers) as ex:
            # submit with controlled prefetch window
            next_to_submit = 0
            inflight: Dict[int, futures.Future] = {}

            def submit_one(pi: int) -> None:
                inflight[pi] = ex.submit(prepare_page, pdf_path, pi, low_dpi, high_dpi, max_crops)

            while next_to_submit < total_pages and len(inflight) < prefetch:
                submit_one(next_to_submit)
                next_to_submit += 1

            next_to_consume = 0
            while next_to_consume < total_pages:
                # Ensure we have future for next_to_consume
                if next_to_consume not in inflight:
                    submit_one(next_to_submit)
                    next_to_submit += 1

                fut = inflight.pop(next_to_consume)
                res: PagePrepResult = fut.result()

                # keep pipeline full
                while next_to_submit < total_pages and len(inflight) < prefetch:
                    submit_one(next_to_submit)
                    next_to_submit += 1

                page_started = time.perf_counter()
                page_deadline = (page_started + page_timeout_s) if page_timeout_s > 0 else None

                # UI diagnostics
                diag = dict(res.diagnostics)
                diag["mode"] = res.mode
                diag["boxes"] = diag.get("boxes", "0")
                diag["pr"] = diag.get("pr_crops", "0")
                diag["hw"] = diag.get("hw_crops", "0")
                if page_timeout_s > 0:
                    diag["page_timeout_s"] = str(page_timeout_s)
                pages_bar.set_postfix(diag)

                # Drift aggregates (page-level)
                agg_mode_counts[res.mode] = agg_mode_counts.get(res.mode, 0) + 1
                try:
                    if "prep_ms" in res.diagnostics:
                        agg_prep_ms.append(float(res.diagnostics["prep_ms"]))
                except Exception:
                    pass
                try:
                    if "boxes" in res.diagnostics:
                        agg_total_boxes.append(float(res.diagnostics["boxes"]))
                except Exception:
                    pass
                try:
                    if "pr_crops" in res.diagnostics:
                        agg_pr_crops.append(float(res.diagnostics["pr_crops"]))
                except Exception:
                    pass
                try:
                    if "hw_crops" in res.diagnostics:
                        agg_hw_crops.append(float(res.diagnostics["hw_crops"]))
                except Exception:
                    pass

                # Write page header in txt
                txt_f.write(f"\n\n=== PAGE {res.page_index + 1} / {total_pages} ({res.mode}) ===\n\n")

                page_records: List[Dict[str, Any]] = []

                if res.error:
                    fatal_errors += 1
                    err_rec = {
                        "ts_utc": now_iso(),
                        "type": "page_error",
                        "run_id": run_id,
                        "doc_id": doc_id,
                        "source_pdf": str(pdf_path),
                        "page": int(res.page_index + 1),
                        "mode": res.mode,
                        "error": res.error,
                        "diagnostics": res.diagnostics,
                    }
                    write_jsonl_line(jf, err_rec)
                    agg_errors.append(err_rec)
                    txt_f.write(f"[error] {res.error}\n")
                    flush_fsync(txt_f)
                    flush_fsync(jf)
                    pages_bar.update(1)
                    next_to_consume += 1
                    continue

                if res.mode == "embedded":
                    txt = (res.embedded_text or "").strip()
                    if txt:
                        txt_f.write(txt + "\n")
                    rec = {
                        "ts_utc": now_iso(),
                        "type": "block",
                        "run_id": run_id,
                        "doc_id": doc_id,
                        "source_pdf": str(pdf_path),
                        "page": int(res.page_index + 1),
                        "mode": "embedded",
                        "block_id": None,
                        "kind": "embedded",
                        "dpi": None,
                        "bbox": None,
                        "script": detect_primary_script(txt),
                        "text": txt,
                        "flags": {"placeholder": False},
                        "features": None,
                        "diagnostics": res.diagnostics,
                    }
                    agg_block_count += 1
                    scr = detect_primary_script(txt)
                    agg_script_counts[scr] = agg_script_counts.get(scr, 0) + 1
                    write_jsonl_line(jf, rec)
                    # page summary
                    summ = {
                        "ts_utc": now_iso(),
                        "type": "page_summary",
                        "run_id": run_id,
                        "doc_id": doc_id,
                        "source_pdf": str(pdf_path),
                        "page": int(res.page_index + 1),
                        "mode": "embedded",
                        "script": detect_primary_script(txt),
                        "text": txt,
                        "diagnostics": res.diagnostics,
                        "latency_ms": int((time.perf_counter() - page_started) * 1000),
                    }
                    write_jsonl_line(jf, summ)
                    flush_fsync(txt_f)
                    flush_fsync(jf)
                    pages_bar.update(1)
                    next_to_consume += 1
                    continue

                pr_crops = [c for c in res.crops if c.kind == "printed"]
                hw_crops = [c for c in res.crops if c.kind == "handwriting"]

                if locals().get('skip_emit', False):
                    # Full-page engine already emitted blocks for this page
                    pr_crops = []
                    hw_crops = []

                # Inference: printed then handwritten
                pr_texts: List[str] = []
                if pr_crops:
                    if page_deadline is not None and time.perf_counter() > page_deadline:
                        jlog('page_timeout', run_id=run_id, doc_id=doc_id, page=int(res.page_index+1), phase='before_printed')
                        pr_crops = []
                    
                    proc, model = trocr.printed(device, dtype)
                    max_tok = max(tokens_for_crop("printed", c.image) for c in pr_crops[: min(8, len(pr_crops))])
                    pr_texts = infer_batch([c.image for c in pr_crops], proc, model, device, dtype, batch_default, max_tok, f"PR p{res.page_index+1}")

                hw_texts: List[str] = []
                if hw_crops:
                    if page_deadline is not None and time.perf_counter() > page_deadline:
                        jlog('page_timeout', run_id=run_id, doc_id=doc_id, page=int(res.page_index+1), phase='before_handwriting')
                        hw_crops = []
                    
                    proc, model = trocr.handwritten(device, dtype)
                    max_tok = max(tokens_for_crop("handwriting", c.image) for c in hw_crops[: min(8, len(hw_crops))])
                    hw_texts = infer_batch([c.image for c in hw_crops], proc, model, device, dtype, batch_default, max_tok, f"HW p{res.page_index+1}")

                def emit(crops: List[CropRecord], texts: List[str]) -> None:
                    nonlocal agg_block_count, agg_placeholder_count, agg_form_blocks
                    for c, t in zip(crops, texts):
                        clean = (t or "").strip()
                        placeholder = is_placeholder_decode(clean)
                        if (not placeholder) and clean:
                            txt_f.write(clean + "\n")
                        rec = {
                            "ts_utc": now_iso(),
                            "type": "block",
                            "run_id": run_id,
                            "doc_id": doc_id,
                            "source_pdf": str(pdf_path),
                            "page": int(res.page_index + 1),
                            "mode": res.mode,
                            "block_id": int(c.crop_id),
                            "kind": c.kind,
                            "dpi": int(c.dpi),
                            "bbox": c.bbox.to_dict(),
                            "script": detect_primary_script(clean),
                            "text": clean,
                            "flags": {"placeholder": bool(placeholder), "is_form": bool(c.is_form)},
                            "features": {"ink_ratio": float(c.ink_ratio), "ink_pixels": int(c.ink_pixels)},
                            "diagnostics": res.diagnostics,
                        }

                        agg_block_count += 1
                        agg_kind_counts[c.kind] = agg_kind_counts.get(c.kind, 0) + 1
                        if placeholder:
                            agg_placeholder_count += 1
                        scr = detect_primary_script(clean)
                        agg_script_counts[scr] = agg_script_counts.get(scr, 0) + 1
                        if c.is_form:
                            agg_form_blocks += 1
                        try:
                            agg_ink_ratios.append(float(c.ink_ratio))
                        except Exception:
                            pass
                        write_jsonl_line(jf, rec)

                if not locals().get('skip_emit', False):
                    emit(pr_crops, pr_texts)
                if not locals().get('skip_emit', False):
                    emit(hw_crops, hw_texts)

                # page summary: concatenation of clean non-placeholder blocks
                # (makes full-text indexing easy)
                # We reconstruct by reading from crop outputs rather than re-parsing txt output.
                all_clean = []
                for t in pr_texts + hw_texts:
                    t = (t or "").strip()
                    if t and (not is_placeholder_decode(t)):
                        all_clean.append(t)
                page_text = "\n".join(all_clean)
                summ = {
                    "ts_utc": now_iso(),
                    "type": "page_summary",
                    "run_id": run_id,
                    "doc_id": doc_id,
                    "source_pdf": str(pdf_path),
                    "page": int(res.page_index + 1),
                    "mode": res.mode,
                    "script": detect_primary_script(page_text),
                    "text": page_text,
                    "diagnostics": res.diagnostics,
                    "latency_ms": int((time.perf_counter() - page_started) * 1000),
                }
                agg_page_latency_ms.append(float(summ.get("latency_ms", 0)))
                write_jsonl_line(jf, summ)

                flush_fsync(txt_f)
                flush_fsync(jf)

                pages_bar.update(1)
                next_to_consume += 1

    except KeyboardInterrupt:
        print("\n[clarity] interrupted", file=sys.stderr)
        return 130
    finally:
        pages_bar.close()
        with contextlib.suppress(Exception):
            txt_f.close()
        with contextlib.suppress(Exception):
            jf.close()

    # -----------------------------
    # Drift metrics + deviation report (auto)
    # -----------------------------
    total_blocks = int(agg_block_count)
    placeholder_rate = _pct(agg_placeholder_count, total_blocks)
    pages_with_errors = int(len(agg_errors))
    error_rate = _pct(pages_with_errors, total_pages)

    metrics = {
        "ts_utc": now_iso(),
        "clarity_version": CLARITY_VERSION,
        "run_id": run_id,
        "doc_id": doc_id,
        "source_pdf": str(pdf_path),
        "doc_sha256": doc_hash,
        "device": device_tag,
        "pages": int(total_pages),
        "pages_with_errors": pages_with_errors,
        "error_rate": error_rate,
        "blocks": total_blocks,
        "placeholders": int(agg_placeholder_count),
        "placeholder_rate": placeholder_rate,
        "mode_counts": agg_mode_counts,
        "kind_counts": agg_kind_counts,
        "script_counts": agg_script_counts,
        "form_blocks": int(agg_form_blocks),
        "timings_ms": {
            "prep_mean": (sum(agg_prep_ms) / len(agg_prep_ms)) if agg_prep_ms else None,
            "prep_p95": _quantile(agg_prep_ms, 0.95),
            "page_latency_mean": (sum(agg_page_latency_ms) / len(agg_page_latency_ms)) if agg_page_latency_ms else None,
            "page_latency_p95": _quantile(agg_page_latency_ms, 0.95),
        },
        "crop_stats": {
            "boxes_mean": (sum(agg_total_boxes) / len(agg_total_boxes)) if agg_total_boxes else None,
            "boxes_p95": _quantile(agg_total_boxes, 0.95),
            "printed_crops_mean": (sum(agg_pr_crops) / len(agg_pr_crops)) if agg_pr_crops else None,
            "handwritten_crops_mean": (sum(agg_hw_crops) / len(agg_hw_crops)) if agg_hw_crops else None,
        },
        "ink": {
            "ink_ratio_mean": (sum(agg_ink_ratios) / len(agg_ink_ratios)) if agg_ink_ratios else None,
            "ink_ratio_p10": _quantile(agg_ink_ratios, 0.10),
            "ink_ratio_p50": _quantile(agg_ink_ratios, 0.50),
            "ink_ratio_p90": _quantile(agg_ink_ratios, 0.90),
        },
    }

    baseline = None
    baseline_created = False
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            baseline = None
    else:
        baseline = metrics
        baseline_created = True
        _write_json(baseline_path, baseline)

    def dist_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
        s = sum(counts.values())
        return {k: (v / s) for k, v in counts.items()} if s else {}

    deviation = {
        "ts_utc": now_iso(),
        "clarity_version": CLARITY_VERSION,
        "run_id": run_id,
        "doc_id": doc_id,
        "baseline_path": str(baseline_path),
        "baseline_created": baseline_created,
        "alerts": [],
        "deltas": {},
    }

    if baseline and (not baseline_created):
        js_script = _js_divergence(dist_from_counts(baseline.get("script_counts", {})), dist_from_counts(metrics.get("script_counts", {})))
        js_mode = _js_divergence(dist_from_counts(baseline.get("mode_counts", {})), dist_from_counts(metrics.get("mode_counts", {})))
        deviation["deltas"]["js_script"] = js_script
        deviation["deltas"]["js_mode"] = js_mode

        b_ph = baseline.get("placeholder_rate")
        c_ph = metrics.get("placeholder_rate")
        if isinstance(b_ph, (int, float)) and isinstance(c_ph, (int, float)):
            deviation["deltas"]["placeholder_rate_abs"] = c_ph - b_ph

        b_er = baseline.get("error_rate")
        c_er = metrics.get("error_rate")
        if isinstance(b_er, (int, float)) and isinstance(c_er, (int, float)):
            deviation["deltas"]["error_rate_abs"] = c_er - b_er

        b_lat = (baseline.get("timings_ms", {}) or {}).get("page_latency_mean")
        c_lat = (metrics.get("timings_ms", {}) or {}).get("page_latency_mean")
        if isinstance(b_lat, (int, float)) and isinstance(c_lat, (int, float)) and b_lat > 0:
            deviation["deltas"]["page_latency_mean_ratio"] = c_lat / b_lat

        # Alerts: simple, reliable thresholds (tune per insurer)
        if metrics["error_rate"] > 0.0:
            deviation["alerts"].append({"id": "P1_PAGE_ERRORS", "why": f"{pages_with_errors} pages failed"})
        if isinstance(b_ph, (int, float)) and c_ph > max(0.08, float(b_ph) * 1.8):
            deviation["alerts"].append({"id": "P1_JUNK_SPIKE", "why": f"placeholder_rate={c_ph:.3f} vs baseline={b_ph:.3f}"})
        if deviation["deltas"].get("page_latency_mean_ratio", 1.0) > 1.5:
            deviation["alerts"].append({"id": "P2_LATENCY_REGRESSION", "why": f"mean page latency up {deviation['deltas']['page_latency_mean_ratio']:.2f}x"})
        if js_script > 0.18:
            deviation["alerts"].append({"id": "P2_SCRIPT_MIX_SHIFT", "why": f"script distribution shift (JS={js_script:.3f})"})
        if js_mode > 0.18:
            deviation["alerts"].append({"id": "P2_PAGE_MODE_SHIFT", "why": f"mode distribution shift (JS={js_mode:.3f})"})

    _write_json(metrics_path, metrics)
    _write_json(deviation_path, deviation)

    print(f"[clarity] metrics: {metrics_path}")
    print(f"[clarity] deviation: {deviation_path}")
    print(f"[clarity] baseline: {baseline_path} (created={baseline_created})\n")

    # Atomic rename temp -> final (best effort)
    txt_tmp.replace(out_path)
    jsonl_tmp.replace(jsonl_path)

    if fatal_errors > 0:
        print(f"[done] completed with {fatal_errors} page errors. outputs written.")
        return 10  # non-zero to signal partial failure
    print("[done] success.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
