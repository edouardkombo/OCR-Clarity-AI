#!/usr/bin/env python3
"""
ocr_clarity.py

Automatic OCR for multi-page PDFs (insurance-friendly).
CPU-first, GPU-auto. Streams output. Shows progress bars.

Clarity v5 adds an automatic "autopilot" tuner because waiting minutes per page is not acceptable.

What autopilot does (no flags required):
- Uses a fast, low-DPI pass to classify the page (embedded / printed / handwriting / form).
- Only re-renders at higher DPI when handwriting is present (accuracy where it matters, speed where it doesn't).
- Detects region explosions (too many crops) and automatically retries segmentation with more aggressive merging.
- Skips empty form fields before OCR (prevents classic placeholder junk like "0 000 000 ...").
- Dynamically adjusts max_new_tokens per crop type (printed vs handwriting) to reduce decoder time.

Usage:
  python ocr_clarity.py input.pdf

Optional:
  python ocr_clarity.py input.pdf --out out.txt
  python ocr_clarity.py input.pdf --dpi 300
  python ocr_clarity.py input.pdf --max-pages 50

Optional environment overrides (still no extra flags needed):
  OCR_WORKERS=4        # page-prep workers (default: min(4, cpu_count))
  OCR_PREFETCH=2       # how many pages to prep ahead (default: 2)

Autopilot knobs (optional):
  OCR_LOW_DPI=200      # low DPI pass for printed/form (default 200)
  OCR_HIGH_DPI=300     # high DPI pass for handwriting (default max(--dpi, 300))
  OCR_MAX_CROPS=180    # if crops exceed this, segmentation retries more aggressively
"""

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

try:
    from concurrent.futures import ThreadPoolExecutor
except Exception as e:
    raise RuntimeError("ThreadPoolExecutor is required for parallel page preparation.") from e


def import_pymupdf():
    """
    Purpose: Import PyMuPDF reliably, avoiding 'fitz' name collisions on Windows.
    Problem it solves: Some environments install a different 'fitz' package with no open().
    Tests to ensure: returned module supports open() and Matrix().
    """
    try:
        import pymupdf as fitz  # type: ignore
        return fitz
    except Exception:
        import fitz  # type: ignore
        return fitz


_TLS = threading.local()

def get_thread_doc(pdf_path: str):
    """
    Purpose: Keep a per-thread PDF handle for safe parallel rendering.
    Problem it solves: Sharing one PyMuPDF doc across threads can be unsafe.
    Tests to ensure: each thread opens the PDF at most once and reuses it.
    """
    fitz_mod = getattr(_TLS, "fitz_mod", None)
    doc = getattr(_TLS, "doc", None)
    cur_path = getattr(_TLS, "pdf_path", None)

    if doc is None or fitz_mod is None or cur_path != pdf_path:
        fitz_mod = import_pymupdf()
        doc = fitz_mod.open(pdf_path)
        _TLS.fitz_mod = fitz_mod
        _TLS.doc = doc
        _TLS.pdf_path = pdf_path

    return fitz_mod, doc


def auto_device() -> torch.device:
    """
    Purpose: Choose the best available device automatically.
    Problem it solves: Uses GPU when present without requiring flags.
    Tests to ensure: CUDA preferred, then MPS, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_dtype(device: torch.device) -> torch.dtype:
    """
    Purpose: Choose a safe precision.
    Problem it solves: fp16 saves VRAM and speeds GPU inference; CPU stays fp32.
    Tests to ensure: CPU is fp32, CUDA/MPS is fp16.
    """
    if device.type == "cpu":
        return torch.float32
    return torch.float16


def render_page_to_pil(fitz_mod, doc, page_index: int, dpi: int) -> Image.Image:
    """
    Purpose: Render one PDF page to RGB image.
    Problem it solves: Scanned PDFs need rasterization for OCR.
    Tests to ensure: renders without alpha and matches expected size.
    """
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz_mod.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def extract_embedded_text(page) -> str:
    """
    Purpose: Extract text layer if present.
    Problem it solves: Avoid OCR when PDF already has selectable text.
    Tests to ensure: returns empty on scanned images, non-empty on digital PDFs.
    """
    try:
        txt = page.get_text("text") or ""
        txt = re.sub(r"[ \t]+\n", "\n", txt).strip()
        return txt
    except Exception:
        return ""


def deskew_binary(bin_img: np.ndarray) -> np.ndarray:
    """
    Purpose: Deskew binarized page.
    Problem it solves: Skew harms segmentation and OCR quality.
    Tests to ensure: small skew corrected, near-straight unchanged.
    """
    coords = np.column_stack(np.where(bin_img < 128))
    if coords.size == 0:
        return bin_img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return bin_img
    h, w = bin_img.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(bin_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_scan(pil_img: Image.Image, do_deskew: bool = True) -> np.ndarray:
    """
    Purpose: Make ink/background consistent.
    Problem it solves: Uneven lighting, compression noise, faint pen strokes.
    Tests to ensure: background uniformity improves; ink remains.
    """
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)

    if float(np.mean(bin_img)) < 127.0:
        bin_img = 255 - bin_img

    if do_deskew:
        bin_img = deskew_binary(bin_img)

    return bin_img


def downscale_for_analysis(bin_img: np.ndarray, target_w: int = 1200) -> np.ndarray:
    """
    Purpose: Create a smaller copy for expensive heuristics.
    Problem it solves: Speeds up per-page checks without affecting OCR crops.
    Tests to ensure: output preserves general structure.
    """
    h, w = bin_img.shape
    if w <= target_w:
        return bin_img
    scale = target_w / float(w)
    nh = max(1, int(h * scale))
    return cv2.resize(bin_img, (target_w, nh), interpolation=cv2.INTER_AREA)


def crop_to_ink(bin_img: np.ndarray, pad: int = 18, min_ink_pixels: int = 250) -> np.ndarray:
    """
    Purpose: Crop whitespace around ink to reduce region count and speed OCR.
    Problem it solves: Sparse handwriting notes often contain huge blank areas.
    Tests to ensure: does not crop away all content; returns original if too little ink.
    """
    inv = 255 - bin_img
    ys, xs = np.where(inv > 0)
    if ys.size < min_ink_pixels:
        return bin_img

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    h, w = bin_img.shape
    y0 = max(0, y0 - pad)
    y1 = min(h - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w - 1, x1 + pad)

    if (y1 - y0) / h > 0.95 and (x1 - x0) / w > 0.95:
        return bin_img

    return bin_img[y0:y1 + 1, x0:x1 + 1]


def grid_line_ratio(bin_img_small: np.ndarray) -> float:
    """
    Purpose: Estimate form/grid presence using long line detection.
    Problem it solves: Forms need different segmentation than free text pages.
    Tests to ensure: forms show higher ratio than handwriting-only pages.
    """
    inv = 255 - bin_img_small
    h, w = inv.shape
    hk = max(20, w // 18)
    vk = max(20, h // 18)
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    lines = cv2.bitwise_or(horiz, vert)
    return float(np.sum(lines > 0)) / float(h * w)


def remove_form_lines(bin_img: np.ndarray) -> np.ndarray:
    """
    Purpose: Remove long horizontal/vertical form lines.
    Problem it solves: Grid lines create bogus segments and confuse OCR.
    Tests to ensure: line pixels reduced; text strokes remain.
    """
    inv = 255 - bin_img
    h, w = inv.shape
    hk = max(20, w // 18)
    vk = max(20, h // 18)
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    lines = cv2.bitwise_or(horiz, vert)
    inv_no = cv2.subtract(inv, lines)
    return 255 - inv_no


def page_is_print_like(bin_img_small: np.ndarray) -> bool:
    """
    Purpose: Detect scanned printed pages that are not forms.
    Problem it solves: Printed scans routed to handwriting OCR are slow and inaccurate.
    Tests to ensure: printed letter scans return True; handwriting pages return False.
    """
    inv = 255 - bin_img_small
    h, w = inv.shape

    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    num, _, stats, _ = cv2.connectedComponentsWithStats((inv > 0).astype(np.uint8), connectivity=8)
    if num <= 2:
        return False

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.int32)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(np.int32)

    keep = (areas >= 10) & (ws >= 2) & (hs >= 2)
    areas = areas[keep]
    hs = hs[keep]

    if areas.size < 30:
        return False

    ink_ratio = float(np.sum(inv > 0)) / float(h * w)
    med_area = float(np.median(areas))
    med_h = float(np.median(hs))
    comp_count = int(areas.size)

    if ink_ratio < 0.01:
        return True
    if comp_count > 150 and med_area < 120 and med_h < 18:
        return True
    if comp_count > 90 and med_area < 100 and med_h < 16:
        return True
    return False


def region_is_print_like(region_bin: np.ndarray) -> bool:
    """
    Purpose: Decide whether a region looks like printed text.
    Problem it solves: Forms contain both printed labels and handwritten fill-ins.
    Tests to ensure: printed regions route to printed model; handwriting to handwritten model.
    """
    inv = 255 - region_bin
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    num, _, stats, _ = cv2.connectedComponentsWithStats((inv > 0).astype(np.uint8), connectivity=8)
    if num <= 2:
        return False

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(np.int32)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.int32)

    keep = (areas >= 12) & (ws >= 2) & (hs >= 2)
    areas = areas[keep]
    hs = hs[keep]

    if areas.size < 10:
        return False

    h, w = region_bin.shape
    ink_ratio = float(np.sum(inv > 0)) / float(h * w)
    med_area = float(np.median(areas))
    med_h = float(np.median(hs))
    comp_count = int(areas.size)

    if ink_ratio < 0.01:
        return True
    if comp_count > 120 and med_area < 110 and med_h < 18:
        return True
    if comp_count > 60 and med_area < 90 and med_h < 16:
        return True
    return False


def segment_projection_lines(bin_img: np.ndarray) -> List[Tuple[int, int]]:
    """
    Purpose: Segment handwriting page into line bands.
    Problem it solves: TrOCR works better on line crops than whole pages.
    Tests to ensure: returns multiple bands on handwriting pages.
    """
    h, w = bin_img.shape
    inv = 255 - bin_img
    inv_dil = cv2.dilate(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)), iterations=1)
    proj = np.sum(inv_dil > 0, axis=1)
    row_thr = max(10, int(0.015 * w))

    raw: List[Tuple[int, int]] = []
    in_band = False
    start = 0
    for y, v in enumerate(proj):
        if (v > row_thr) and (not in_band):
            in_band = True
            start = y
        elif (v <= row_thr) and in_band:
            end = y
            if end - start >= 12:
                raw.append((start, end))
            in_band = False
    if in_band:
        end = h - 1
        if end - start >= 12:
            raw.append((start, end))

    merged: List[Tuple[int, int]] = []
    pad = 6
    for s, e in raw:
        s = max(0, s - pad)
        e = min(h, e + pad)
        if merged and s - merged[-1][1] < 6:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    def pad(self, px: int, py: int, w: int, h: int) -> "Box":
        return Box(max(0, self.x0 - px), max(0, self.y0 - py), min(w, self.x1 + px), min(h, self.y1 + py))


def sort_boxes_reading_order(boxes: List[Box], y_tol: int) -> List[Box]:
    """
    Purpose: Sort regions in reading order.
    Problem it solves: Contours come unsorted.
    Tests to ensure: stable top-to-bottom, left-to-right ordering.
    """
    if not boxes:
        return boxes
    boxes_sorted = sorted(boxes, key=lambda b: (b.y0, b.x0))
    lines: List[List[Box]] = []
    cur: List[Box] = [boxes_sorted[0]]
    for b in boxes_sorted[1:]:
        if abs(b.y0 - cur[-1].y0) <= y_tol:
            cur.append(b)
        else:
            lines.append(sorted(cur, key=lambda x: x.x0))
            cur = [b]
    lines.append(sorted(cur, key=lambda x: x.x0))
    out: List[Box] = []
    for ln in lines:
        out.extend(ln)
    return out


def segment_cc_regions(bin_img: np.ndarray, mode: str, kx_mul: float = 1.0, min_area_mul: float = 1.0) -> List[Box]:
    """
    Purpose: Segment page into text regions via connected components.
    Problem it solves: Projection segmentation fails on printed pages, multi-column, and forms.
    Tests to ensure: printed yields fewer boxes, forms yield filled boxes.
    """
    h, w = bin_img.shape
    inv = 255 - bin_img

    if mode == "printed":
        kx = int(max(28, w // 80) * kx_mul)
        ky = 3
    else:
        kx = int(max(18, w // 110) * kx_mul)
        ky = 3

    kx = max(10, min(kx, max(10, w // 2)))

    blob = cv2.dilate(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)), iterations=1)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Box] = []
    base_min_area = max(250, (h * w) // 50000)
    min_area = int(base_min_area * min_area_mul)

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < min_area:
            continue
        if bw > int(0.97 * w) and bh > int(0.97 * h):
            continue
        if bh < 12:
            continue
        boxes.append(Box(x, y, x + bw, y + bh))

    boxes = [b.pad(px=10, py=6, w=w, h=h) for b in boxes]
    y_tol = max(18, h // 120)
    return sort_boxes_reading_order(boxes, y_tol=y_tol)


def region_has_enough_ink(region_bin: np.ndarray, min_ratio: float, min_pixels: int) -> bool:
    """
    Purpose: Skip OCR on empty boxes/whitespace regions.
    Problem it solves: Empty form fields often decode as placeholders.
    Tests to ensure: blank boxes are skipped; real text areas are kept.
    """
    inv = 255 - region_bin
    ink_pixels = int(np.count_nonzero(inv))
    if ink_pixels < min_pixels:
        return False
    ratio = ink_pixels / float(region_bin.size)
    return ratio >= min_ratio


_PLACEHOLDER_RE_1 = re.compile(r"^(?:0\s*){10,}$")
_PLACEHOLDER_RE_2 = re.compile(r"^(?:0\s*000\s*){2,}0?$")
_PLACEHOLDER_RE_3 = re.compile(r"^(?:#\s*){6,}$")

def is_placeholder_decode(s: str) -> bool:
    """
    Purpose: Prevent polluted lines from appearing in output.
    Problem it solves: OCR on empty/line-only crops produces junk like '1 000 000 ...' or repeated phrases.
    Tests to ensure:
      - drops long 000-chains and symbol soup
      - keeps normal numbers, IDs, and real sentences
    """
    t = (s or "").strip()
    if not t:
        return True

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return True

    compact = re.sub(r"\s+", "", t)
    if not compact:
        return True

    # 1) Pure symbol junk
    if re.fullmatch(r"[#_\-\=\|\.\,\:\;\/\+\*\(\)\[\]]{3,}", compact):
        return True

    # 2) Very low character diversity on long strings (classic blank-field hallucination)
    if len(compact) >= 40:
        uniq = set(compact.lower())
        if len(uniq) <= 3:
            return True

    # 3) Long chains of 000 groups (common on empty boxes/lines)
    # Examples: "1 000 000 000 000 ..." or "2 000 000 ..."
    if len(t) >= 30 and re.fullmatch(r"\d(?:\s*000){8,}\s*\d?", t):
        return True

    # 4) All digits with spaces where one token dominates (0/00/000/0000)
    tokens = t.split(" ")
    if len(tokens) >= 10:
        from collections import Counter
        c = Counter(tokens)
        tok, freq = c.most_common(1)[0]
        if freq / float(len(tokens)) >= 0.8 and tok in {"0", "00", "000", "0000"}:
            return True

    # 5) Alphabet soup like "a b c d e f g h i"
    if re.fullmatch(r"(?:[A-Za-z]\s+){8,}[A-Za-z]", t):
        return True

    # 6) Repetition loops like "of the of the of the..." or similar
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

    # 7) Original strict zero-only filters
    if " " in t and re.fullmatch(r"0+", compact) and len(compact) >= 6:
        return True
    if re.fullmatch(r"(?:0|#)+", compact) and len(compact) >= 8:
        return True
    if _PLACEHOLDER_RE_1.fullmatch(t):
        return True
    if _PLACEHOLDER_RE_2.fullmatch(t):
        return True
    if _PLACEHOLDER_RE_3.fullmatch(t):
        return True

    return False


class TrOCRBundle:
    """
    Purpose: Lazily load handwritten and printed TrOCR models only when needed.
    Problem it solves: Keeps memory low while supporting mixed documents.
    Tests to ensure: printed loads only if routed; models placed on correct device/dtype.
    """
    def __init__(self, device: torch.device, dtype: torch.dtype, cache_dir: Optional[str]):
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self._hw = None
        self._pr = None

    def _load(self, name: str):
        processor = TrOCRProcessor.from_pretrained(name, cache_dir=self.cache_dir, use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained(name, cache_dir=self.cache_dir)
        if self.device.type != "cpu":
            model = model.to(device=self.device, dtype=self.dtype)
        else:
            model = model.to(device=self.device)
        model.eval()
        return processor, model

    def handwritten(self):
        if self._hw is None:
            self._hw = self._load("microsoft/trocr-small-handwritten")
        return self._hw

    def printed(self):
        if self._pr is None:
            self._pr = self._load("microsoft/trocr-small-printed")
        return self._pr


def tokens_for_crop(kind: str, img: Image.Image) -> int:
    """
    Purpose: Keep generation short when the line is short.
    Problem it solves: The TrOCR decoder cost grows with max_new_tokens.
    Tests to ensure: printed crops use fewer tokens than handwriting; big crops get more tokens.
    """
    w, h = img.size
    aspect = w / float(max(1, h))
    if kind == "printed":
        if aspect > 8.0 or w > 1400:
            return 64
        return 48
    if aspect > 7.0 or w > 1400 or h > 260:
        return 80
    return 64


def infer_with_retry(
    images: List[Image.Image],
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    max_new_tokens: int,
    pbar_desc: str,
) -> List[str]:
    """
    Purpose: Run inference with automatic batch-size backoff on OOM.
    Problem it solves: GPU VRAM varies; keeps it automatic.
    Tests to ensure: if OOM occurs, batch size reduces and continues.
    """
    out: List[str] = []
    if not images:
        return out

    bs = max(1, int(batch_size))
    while True:
        try:
            steps = range(0, len(images), bs)
            pbar = tqdm(steps, desc=pbar_desc, unit="batch", leave=False)
            for i in pbar:
                batch = images[i:i + bs]
                inputs = processor(images=batch, return_tensors="pt")
                px = inputs.pixel_values
                if device.type != "cpu":
                    px = px.to(device=device, dtype=dtype, non_blocking=True)
                else:
                    px = px.to(device=device)

                with torch.inference_mode():
                    gen = model.generate(px, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, no_repeat_ngram_size=3, repetition_penalty=1.12)

                txt = processor.batch_decode(gen, skip_special_tokens=True)
                out.extend([t.strip() for t in txt])
            return out
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and bs > 1:
                bs = max(1, bs // 2)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise


@dataclass
class PagePrepResult:
    page_index: int
    mode: str
    embedded_text: Optional[str]
    printed_crops: List[Image.Image]
    handwritten_crops: List[Image.Image]
    diagnostics: Dict[str, str]


def classify_mode_from_low(bin_low: np.ndarray) -> Tuple[str, Dict[str, str]]:
    bin_small = downscale_for_analysis(bin_low, target_w=1200)
    lr = grid_line_ratio(bin_small)
    bands_small = segment_projection_lines(bin_small)
    bands_count = len(bands_small)
    avg_band_h = (sum((e - s) for s, e in bands_small) / float(bands_count)) if bands_count else 0.0

    is_form = (lr >= 0.015) or (bands_count >= 140) or (avg_band_h < 18 and bands_count > 40)
    if is_form:
        mode = "form"
    else:
        mode = "printed" if page_is_print_like(bin_small) else "handwriting"

    diag = {"mode": mode, "grid": f"{lr:.3f}", "bands": str(bands_count)}
    return mode, diag


def autopilot_segment_boxes(work_low: np.ndarray, mode: str, max_crops: int) -> Tuple[List[Box], int]:
    tries = [
        (1.00, 1.00),
        (1.35, 1.10),
        (1.70, 1.25),
    ]
    best_boxes: List[Box] = []
    best_count = 10**9

    for kx_mul, area_mul in tries:
        boxes = segment_cc_regions(work_low, mode=mode, kx_mul=kx_mul, min_area_mul=area_mul)
        c = len(boxes)
        if c < best_count:
            best_boxes, best_count = boxes, c
        if c <= max_crops:
            return boxes, c

    return best_boxes, best_count


def prepare_page(pdf_path: str, page_index: int, user_dpi: int) -> PagePrepResult:
    t0 = time.perf_counter()
    fitz_mod, doc = get_thread_doc(pdf_path)
    page = doc.load_page(page_index)

    embedded = extract_embedded_text(page)
    embedded_compact = re.sub(r"\s+", "", embedded)
    if len(embedded_compact) >= 25:
        return PagePrepResult(page_index, "embedded", embedded, [], [], {"mode": "embedded", "prep_ms": f"{(time.perf_counter()-t0)*1000:.0f}"})

    low_dpi = int(os.environ.get("OCR_LOW_DPI", "200"))
    high_dpi = int(os.environ.get("OCR_HIGH_DPI", str(max(int(user_dpi), 300))))
    max_crops = int(os.environ.get("OCR_MAX_CROPS", "180"))

    pil_low = render_page_to_pil(fitz_mod, doc, page_index, low_dpi)
    bin_low = preprocess_scan(pil_low, do_deskew=True)
    mode, diag = classify_mode_from_low(bin_low)

    printed_crops: List[Image.Image] = []
    handwritten_crops: List[Image.Image] = []

    if mode == "handwriting":
        pil_hi = render_page_to_pil(fitz_mod, doc, page_index, high_dpi)
        bin_hi = preprocess_scan(pil_hi, do_deskew=True)
        work = crop_to_ink(bin_hi, pad=18, min_ink_pixels=220)
        bands = segment_projection_lines(work) or [(0, work.shape[0])]
        for (y0, y1) in bands:
            reg = work[y0:y1, 0:work.shape[1]]
            if not region_has_enough_ink(reg, min_ratio=float(os.environ.get("OCR_HW_MIN_RATIO", "0.0022")), min_pixels=int(os.environ.get("OCR_HW_MIN_PIXELS", "140"))):
                continue
            handwritten_crops.append(Image.fromarray(reg).convert("RGB"))
        diag["hi_dpi"] = str(high_dpi)
        diag["hw_crops"] = str(len(handwritten_crops))
        diag["prep_ms"] = f"{(time.perf_counter()-t0)*1000:.0f}"
        return PagePrepResult(page_index, mode, None, [], handwritten_crops, diag)

    work_low = remove_form_lines(bin_low) if mode == "form" else bin_low
    boxes, boxes_count = autopilot_segment_boxes(work_low, mode=mode, max_crops=max_crops)
    diag["boxes"] = str(boxes_count)
    if not boxes:
        boxes = [Box(0, 0, work_low.shape[1], work_low.shape[0])]

    regions_bin_low: List[np.ndarray] = []
    regions_box_low: List[Box] = []
    for b in boxes:
        reg = work_low[b.y0:b.y1, b.x0:b.x1]
        if reg.size == 0:
            continue
        if not region_has_enough_ink(reg, min_ratio=float(os.environ.get("OCR_PR_MIN_RATIO", "0.0018")) if mode == "printed" else float(os.environ.get("OCR_FORM_MIN_RATIO", "0.0020")), min_pixels=int(os.environ.get("OCR_PR_MIN_PIXELS", "160")) if mode == "printed" else int(os.environ.get("OCR_FORM_MIN_PIXELS", "180"))):
            continue
        regions_bin_low.append(reg)
        regions_box_low.append(b)

    if mode == "printed":
        printed_crops = [Image.fromarray(reg).convert("RGB") for reg in regions_bin_low]
        diag["pr_crops"] = str(len(printed_crops))
        diag["prep_ms"] = f"{(time.perf_counter()-t0)*1000:.0f}"
        return PagePrepResult(page_index, mode, None, printed_crops, [], diag)

    pr_boxes: List[Box] = []
    hw_boxes: List[Box] = []
    for b, reg_low in zip(regions_box_low, regions_bin_low):
        h, w = reg_low.shape
        if w > 3 * h:
            pr_boxes.append(b)
        else:
            pr_boxes.append(b) if region_is_print_like(reg_low) else hw_boxes.append(b)

    for b in pr_boxes:
        reg = work_low[b.y0:b.y1, b.x0:b.x1]
        printed_crops.append(Image.fromarray(reg).convert("RGB"))

    if hw_boxes:
        pil_hi = render_page_to_pil(fitz_mod, doc, page_index, high_dpi)
        bin_hi = preprocess_scan(pil_hi, do_deskew=True)
        work_hi = remove_form_lines(bin_hi)
        scale = high_dpi / float(low_dpi)
        hH, wH = work_hi.shape
        for b in hw_boxes:
            x0 = int(b.x0 * scale); y0 = int(b.y0 * scale)
            x1 = int(b.x1 * scale); y1 = int(b.y1 * scale)
            x0 = max(0, min(wH - 1, x0)); x1 = max(0, min(wH, x1))
            y0 = max(0, min(hH - 1, y0)); y1 = max(0, min(hH, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            reg = work_hi[y0:y1, x0:x1]
            if not region_has_enough_ink(reg, min_ratio=float(os.environ.get("OCR_FORM_HW_MIN_RATIO", "0.0022")), min_pixels=int(os.environ.get("OCR_FORM_HW_MIN_PIXELS", "160"))):
                continue
            handwritten_crops.append(Image.fromarray(reg).convert("RGB"))
        diag["hi_dpi"] = str(high_dpi)

    diag["pr_crops"] = str(len(printed_crops))
    diag["hw_crops"] = str(len(handwritten_crops))
    diag["prep_ms"] = f"{(time.perf_counter()-t0)*1000:.0f}"
    return PagePrepResult(page_index, mode, None, printed_crops, handwritten_crops, diag)


def main():
    ap = argparse.ArgumentParser(description="Automatic OCR for insurance PDFs (Clarity v5: autopilot speed).")
    ap.add_argument("pdf", type=str, help="Input PDF (scanned or digital)")
    ap.add_argument("--out", type=str, default=None, help="Output text path (default: input_basename.txt)")
    ap.add_argument("--dpi", type=int, default=300, help="Target high DPI for handwriting (default 300)")
    ap.add_argument("--max-pages", type=int, default=None, help="Optional safety cap for huge PDFs")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {pdf_path}")

    out_path = Path(args.out) if args.out else pdf_path.with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = auto_device()
    dtype = auto_dtype(device)

    batch_default = 2 if device.type == "cpu" else 8

    cpu_cnt = os.cpu_count() or 4
    default_workers = max(1, min(4, cpu_cnt))
    workers = int(os.environ.get("OCR_WORKERS", str(default_workers)))
    prefetch = int(os.environ.get("OCR_PREFETCH", "2"))
    prefetch = max(1, min(8, prefetch))
    workers = max(1, min(16, workers))

    if workers > 1:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    try:
        if device.type == "cpu":
            torch.set_num_threads(max(1, cpu_cnt))
    except Exception:
        pass

    fitz_mod = import_pymupdf()
    doc0 = fitz_mod.open(str(pdf_path))
    total_pages = doc0.page_count
    if args.max_pages is not None:
        total_pages = min(total_pages, int(args.max_pages))
    doc0.close()

    low_dpi = int(os.environ.get("OCR_LOW_DPI", "200"))
    high_dpi = int(os.environ.get("OCR_HIGH_DPI", str(max(int(args.dpi), 300))))
    max_crops = int(os.environ.get("OCR_MAX_CROPS", "180"))

    print(f"[info] Device: {device.type} | dtype: {str(dtype).replace('torch.', '')} | pages: {total_pages}")
    print(f"[info] Autopilot: low_dpi={low_dpi} high_dpi={high_dpi} max_crops={max_crops}")
    print(f"[info] Page-prep parallelism: workers={workers} prefetch={prefetch}")

    trocr = TrOCRBundle(device=device, dtype=dtype, cache_dir=None)

    futures: Dict[int, object] = {}
    next_submit = 0

    def submit_one(i: int):
        futures[i] = executor.submit(prepare_page, str(pdf_path), i, int(high_dpi))

    with ThreadPoolExecutor(max_workers=workers) as executor, out_path.open("w", encoding="utf-8") as f:
        while next_submit < total_pages and len(futures) < prefetch:
            submit_one(next_submit)
            next_submit += 1

        pages_bar = tqdm(range(total_pages), desc="Pages", unit="page")

        for pi in pages_bar:
            if pi not in futures:
                submit_one(pi)

            res: PagePrepResult = futures[pi].result()
            del futures[pi]

            while next_submit < total_pages and len(futures) < prefetch:
                submit_one(next_submit)
                next_submit += 1

            pages_bar.set_postfix({
                "mode": res.diagnostics.get("mode", res.mode),
                "grid": res.diagnostics.get("grid", ""),
                "boxes": res.diagnostics.get("boxes", ""),
                "pr": res.diagnostics.get("pr_crops", ""),
                "hw": res.diagnostics.get("hw_crops", ""),
                "prep_ms": res.diagnostics.get("prep_ms", ""),
            })

            f.write(f"\n\n=== PAGE {res.page_index + 1} / {total_pages} ({res.mode}) ===\n\n")

            if res.mode == "embedded":
                f.write((res.embedded_text or "") + "\n")
                f.flush()
                continue

            if res.printed_crops:
                proc, model_pr = trocr.printed()
                max_tok = max(tokens_for_crop("printed", img) for img in res.printed_crops[: min(8, len(res.printed_crops))])
                pr_txt = infer_with_retry(res.printed_crops, proc, model_pr, device, dtype, batch_default, max_tok, f"PR regions p{res.page_index + 1}")
                for t in pr_txt:
                    if not is_placeholder_decode(t):
                        f.write(t + "\n")

            if res.handwritten_crops:
                proc, model_hw = trocr.handwritten()
                max_tok = max(tokens_for_crop("handwriting", img) for img in res.handwritten_crops[: min(8, len(res.handwritten_crops))])
                hw_txt = infer_with_retry(res.handwritten_crops, proc, model_hw, device, dtype, batch_default, max_tok, f"HW regions p{res.page_index + 1}")
                for t in hw_txt:
                    if not is_placeholder_decode(t):
                        f.write(t + "\n")

            f.flush()

    print(f"[done] Wrote: {out_path}")


if __name__ == "__main__":
    main()
