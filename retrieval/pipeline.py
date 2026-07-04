"""Shelf-analysis pipeline: YOLO detection + SWIN/FAISS retrieval classification.

Reusable core behind the Streamlit dashboard (frontend/app.py) and any future FastAPI backend.
`analyze_image()` runs our fine-tuned YOLO detector (detection/artifacts/v11/best.pt) over a
shelf image, crops every detected product, classifies each crop via SWIN+FAISS retrieval, and
returns an annotated image + a per-product detections table + shelf-level metrics.

Model loads are process-cached so the heavy assets (2 GB FAISS index, SWIN weights, YOLO
weights) are read from disk only once.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
# Reuse our own fine-tuned detector (mAP@0.5 ~0.92); override with $YOLO_WEIGHTS.
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", str(REPO_ROOT / "detection" / "artifacts" / "v11" / "best.pt"))

_INVALID_LABELS = {"unknown", "n/a", "na", "none", ""}


@dataclass
class Detection:
    crop_id: int
    box: tuple[int, int, int, int]
    category: str
    subcategory: str
    score: float  # FAISS L2 distance to nearest neighbour (lower = closer)
    area: int


@dataclass
class AnalysisResult:
    annotated_image: Image.Image
    detections: list[Detection]
    num_items: int
    distinct_categories: int
    empty_pct: float
    empty_label: str
    shelf_type: str
    review_count: int
    timings: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def get_yolo():
    from ultralytics import YOLO

    return YOLO(YOLO_WEIGHTS)


@lru_cache(maxsize=1)
def get_classifier():
    from retrieval.swin_faiss import load_swin_faiss_classifier

    return load_swin_faiss_classifier()


def classifier_ready() -> bool:
    try:
        return get_classifier().is_ready()
    except Exception:
        return False


def _pad_box(x1, y1, x2, y2, img_w, img_h, pad=12):
    return (max(0, int(x1 - pad)), max(0, int(y1 - pad)),
            min(img_w, int(x2 + pad)), min(img_h, int(y2 + pad)))


def _estimate_empty_space(boxes, img_w, img_h, max_dim=640):
    """Fraction of shelf area NOT covered by any detection box."""
    if len(boxes) == 0:
        return 1.0
    scale = max(1, max(img_w, img_h) // max_dim)
    h, w = max(1, img_h // scale), max(1, img_w // scale)
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v / scale)) for v in box[:4]]
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return 1.0 - float(mask.sum()) / (w * h)


def _valid_label(label: str) -> bool:
    if not label:
        return False
    low = label.strip().lower()
    if low in _INVALID_LABELS:
        return False
    if low.startswith(("train ", "train_", "crop ")) or low.endswith(" crop"):
        return False
    return True


def _empty_label(pct: float) -> str:
    return "High" if pct >= 0.55 else "Moderate" if pct >= 0.25 else "Low"


def analyze_image(image: Image.Image, conf: float = 0.25, max_crops: int = 0,
                  pad: int = 12, top_k: int = 10) -> AnalysisResult:
    """Detect + classify products on a shelf image. max_crops=0 classifies every box."""
    import time

    image = image.convert("RGB")
    img_w, img_h = image.size

    t0 = time.time()
    det = get_yolo()(image, conf=conf, verbose=False)
    boxes = det[0].boxes.xyxy.cpu().numpy()
    t_yolo = time.time()

    clf = get_classifier()
    ready = clf.is_ready()

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    order = list(range(len(boxes)))
    if max_crops and max_crops > 0:
        order = order[:max_crops]

    detections: list[Detection] = []
    for i in order:
        x1, y1, x2, y2 = _pad_box(*boxes[i][:4], img_w, img_h, pad=pad)
        crop = image.crop((x1, y1, x2, y2))
        category, subcategory, score = "unknown", "unknown", 0.0
        if ready:
            try:
                r = clf.classify(crop, top_k=top_k, top_labels=5)
                cat = r.get("predicted_category") or r.get("label")
                sub = r.get("predicted_subcategory") or r.get("best_subcategory")
                if _valid_label(cat):
                    category = cat
                    score = float(r.get("score", 0.0) or 0.0)
                    if _valid_label(sub):
                        subcategory = sub
            except Exception:
                pass

        color = "#e5484d" if category == "unknown" else "#30a46c"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), category if category != "unknown" else "?",
                  fill=color, font=font)

        detections.append(Detection(
            crop_id=i + 1, box=(x1, y1, x2, y2), category=category,
            subcategory=subcategory, score=round(score, 3),
            area=int((x2 - x1) * (y2 - y1)),
        ))

    empty_pct = _estimate_empty_space(boxes, img_w, img_h)
    known = [d for d in detections if d.category != "unknown"]
    distinct = len({d.category for d in known})
    review = sum(1 for d in detections if d.category == "unknown")
    shelf_type = "Mixed" if distinct > 1 else "Category-specific" if distinct == 1 else "Unknown"

    return AnalysisResult(
        annotated_image=annotated, detections=detections, num_items=len(detections),
        distinct_categories=distinct, empty_pct=round(empty_pct, 4),
        empty_label=_empty_label(empty_pct), shelf_type=shelf_type, review_count=review,
        timings={"yolo_s": round(t_yolo - t0, 3),
                 "classify_s": round(time.time() - t_yolo, 3), "boxes": len(boxes)},
    )


def detections_to_records(result: AnalysisResult) -> list[dict]:
    return [
        {"crop_id": d.crop_id, "category": d.category, "subcategory": d.subcategory,
         "score": d.score, "area": d.area, "box": list(d.box)}
        for d in result.detections
    ]
