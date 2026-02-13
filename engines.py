#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Engines (auto-router friendly)

Purpose
- Provide pluggable OCR engines for Clarity without changing the output contract.
- Engines are OPTIONAL: if a library is not installed, the engine is unavailable and the router falls back.

Engines
- PaddleOCR: strong multilingual printed OCR + built-in detection. Best for forms/printed pages.
- docTR: strong DL OCR pipeline (detection+recognition). Good fallback if Paddle isn't installed.
- TrOCR (in core CLI): used for handwriting-heavy crops (high accuracy on handwriting when cropped well).

Tests to ensure:
- engine availability detection does not crash when dependencies are missing
- engines return blocks with bbox + text in a consistent schema
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

@dataclass
class OcrBlock:
    # bbox in pixels: [x0, y0, x1, y1]
    bbox: List[int]
    text: str
    conf: Optional[float] = None
    engine: str = "unknown"


def _as_rgb_np(pil_img) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return arr


def is_paddle_available() -> bool:
    try:
        import paddleocr  # noqa
        return True
    except Exception:
        return False


def is_doctr_available() -> bool:
    try:
        import doctr  # noqa
        return True
    except Exception:
        return False


def ocr_fullpage_paddle(pil_img, lang: str = "en") -> List[OcrBlock]:
    """
    Purpose: Full-page OCR with detection using PaddleOCR.
    Problem it solves: Printed forms and multilingual blocks without manual cropping.
    Tests to ensure:
      - returns non-empty blocks on printed pages
      - bbox ordering is sane (x0<x1, y0<y1)
    """
    from paddleocr import PaddleOCR  # type: ignore

    # PaddleOCR initialization is heavy; caller should cache instance when possible.
    # We create a lightweight instance per call only if caching is not provided.
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)  # CPU by default
    img = _as_rgb_np(pil_img)

    res = ocr.ocr(img, cls=True)
    blocks: List[OcrBlock] = []
    # res is list[lines]; for one image: res[0] is list of [points, (text, conf)]
    lines = res[0] if res and isinstance(res, list) else []
    for line in lines:
        try:
            pts, (txt, conf) = line
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, x1 = int(min(xs)), int(max(xs))
            y0, y1 = int(min(ys)), int(max(ys))
            if txt and (x1 > x0) and (y1 > y0):
                blocks.append(OcrBlock(bbox=[x0, y0, x1, y1], text=str(txt), conf=float(conf), engine="paddleocr"))
        except Exception:
            continue
    return blocks


def ocr_fullpage_doctr(pil_img) -> List[OcrBlock]:
    """
    Purpose: Full-page OCR with detection using docTR.
    Problem it solves: Strong fallback pipeline when PaddleOCR is unavailable.
    Tests to ensure:
      - returns blocks on printed pages
      - does not crash if torch/cpu only
    """
    from doctr.models import ocr_predictor  # type: ignore

    predictor = ocr_predictor(pretrained=True)  # CPU unless torch selects otherwise
    doc = predictor([_as_rgb_np(pil_img)])
    blocks: List[OcrBlock] = []
    # docTR returns geometry in relative coords (0..1)
    h, w = pil_img.size[1], pil_img.size[0]
    for page in doc.pages:
        for block in page.blocks:
            for line in block.lines:
                txt = " ".join([wrd.value for wrd in line.words]).strip()
                if not txt:
                    continue
                # line.geometry: ((x0,y0),(x1,y1)) in relative
                (x0r, y0r), (x1r, y1r) = line.geometry
                x0 = int(x0r * w); x1 = int(x1r * w)
                y0 = int(y0r * h); y1 = int(y1r * h)
                if x1 > x0 and y1 > y0:
                    blocks.append(OcrBlock(bbox=[x0, y0, x1, y1], text=txt, conf=None, engine="doctr"))
    return blocks


def guess_paddle_lang(script: str) -> str:
    """
    Map primary script to PaddleOCR language packs.
    Conservative mapping: better to pick a broad model than the wrong narrow one.
    """
    script = (script or "latin").lower()
    if script in {"arabic"}:
        return "ar"
    if script in {"cyrillic"}:
        return "ru"
    if script in {"cjk"}:
        return "ch"
    # latin and unknown -> english by default (works ok for many latin languages)
    return "en"
