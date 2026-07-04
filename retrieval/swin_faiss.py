"""SWIN + FAISS retrieval classifier.

Instead of training a softmax classifier over thousands of product classes (which collapses on
rare classes — the near-zero result the team hit), this embeds each product crop with a SWIN
transformer and looks it up in a FAISS index of ~692k pre-embedded, labeled crops. The nearest
neighbour's category/subcategory becomes the prediction. Retrieval degrades gracefully with many
rare classes, which is why it outperforms the classifiers here.

Adapted from the teammate's repo (Richa-max/dense-shelf-images-object-detection) with:
  - explicit, configurable asset paths (no fragile CWD `*.csv` globbing),
  - the index -> image_path map and image_path -> label map loaded deterministically.

Assets (see retrieval/assets/, provisioned per retrieval/README.md):
  - swin_model_assets/, swin_processor_assets/  : the SWIN encoder + preprocessor
  - swin_faiss_index.bin                        : the FAISS index (query target)
  - swin_faiss_indexed_image_paths.csv          : row i -> the image_path at index id i
  - labels/train_product_category_58.csv        : image_path -> predicted_category/subcategory
"""
from __future__ import annotations

import os

# torch + faiss both bundle libomp on macOS; permit the duplicate so the process doesn't abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

try:
    import faiss
except ImportError:
    faiss = None

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _infer_label_from_path(path: str) -> str:
    """Fallback label when a crop has no metadata row (usually means 'unknown')."""
    name = os.path.splitext(os.path.basename(path.replace("\\", "/")))[0]
    name = name.replace("_", " ").replace("-", " ").strip()
    low = name.lower()
    if low.startswith("train ") or " crop " in low or low.endswith(" crop"):
        return "unknown"
    return name or "unknown"


class SwinFaissClassifier:
    def __init__(
        self,
        model_dir: str | Path = ASSETS_DIR / "swin_model_assets",
        processor_dir: str | Path = ASSETS_DIR / "swin_processor_assets",
        index_path: str | Path = ASSETS_DIR / "swin_faiss_index.bin",
        image_paths_path: str | Path = ASSETS_DIR / "swin_faiss_indexed_image_paths.csv",
        labels_csv: str | Path = ASSETS_DIR / "labels" / "train_product_category_58.csv",
    ):
        self.model_dir = str(model_dir)
        self.processor_dir = str(processor_dir)
        self.index_path = str(index_path)
        self.image_paths_path = str(image_paths_path)
        self.labels_csv = str(labels_csv)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = None
        self.model = None
        self.index = None
        self.image_paths: list[dict] = []
        self.is_ready_flag = False
        self._load_resources()

    def is_ready(self) -> bool:
        return self.is_ready_flag

    # -- loading -----------------------------------------------------------
    def _load_resources(self):
        if faiss is None:
            print("[swin_faiss] faiss not installed (pip install faiss-cpu)")
            return
        missing = [p for p in (self.model_dir, self.processor_dir, self.index_path)
                   if not os.path.exists(p)]
        if missing:
            print(f"[swin_faiss] missing asset paths: {missing}")
            return
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.processor_dir)
            self.model = AutoModel.from_pretrained(self.model_dir)
            self.model.eval().to(self.device)
            self.index = faiss.read_index(self.index_path)
            self.image_paths = self._load_paths()
            self.is_ready_flag = bool(self.index is not None and self.image_paths)
            print(f"[swin_faiss] ready={self.is_ready_flag} "
                  f"(index size={getattr(self.index, 'ntotal', '?')}, "
                  f"paths={len(self.image_paths)})")
        except Exception as exc:
            print(f"[swin_faiss] failed to load resources: {exc}")
            self.is_ready_flag = False

    def _load_label_map(self) -> dict[str, dict]:
        """basename -> {category, subcategory} from the labels CSV."""
        label_map: dict[str, dict] = {}
        if not os.path.exists(self.labels_csv):
            print(f"[swin_faiss] labels CSV not found: {self.labels_csv} (labels -> 'unknown')")
            return label_map
        with open(self.labels_csv, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                img = (row.get("image_path") or "").strip()
                if not img:
                    continue
                cat = (row.get("predicted_category") or row.get("full_label") or "").strip()
                sub = (row.get("predicted_subcategory") or "").strip() or None
                label_map[os.path.basename(img)] = {"label": cat, "subcategory": sub}
        return label_map

    def _load_paths(self) -> list[dict]:
        """Index row order -> {path, label, subcategory}."""
        label_map = self._load_label_map()
        entries: list[dict] = []
        with open(self.image_paths_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            has_col = reader.fieldnames and "image_path" in [f.strip() for f in reader.fieldnames]
            if has_col:
                rows = (str(r.get("image_path", "")).strip() for r in reader)
            else:
                fh.seek(0)
                rows = (line.strip() for line in fh if line.strip().lower() != "image_path")
            for path in rows:
                if not path:
                    continue
                meta = label_map.get(os.path.basename(path))
                if meta and meta.get("label"):
                    entries.append({"path": path, "label": meta["label"],
                                    "subcategory": meta.get("subcategory")})
                else:
                    entries.append({"path": path, "label": _infer_label_from_path(path),
                                    "subcategory": None})
        return entries

    # -- inference ---------------------------------------------------------
    def _embed_image(self, image: Image.Image) -> np.ndarray:
        # Mean-pool the SWIN last_hidden_state (matches how the index was built — raw,
        # non-normalized vectors), returned as float32 for FAISS.
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state.mean(dim=1)
        return hidden.cpu().numpy().astype("float32")

    def query(self, image: Image.Image, top_k: int = 10) -> list[dict]:
        if not self.is_ready():
            raise RuntimeError("Swin FAISS classifier is not ready")
        distances, indices = self.index.search(self._embed_image(image), top_k)
        neighbors = []
        for idx, score in zip(indices[0].tolist(), distances[0].tolist()):
            if 0 <= idx < len(self.image_paths):
                d = self.image_paths[idx]
                neighbors.append({"path": d["path"], "label": d["label"],
                                  "subcategory": d.get("subcategory"), "score": float(score)})
        return neighbors

    def classify(self, image: Image.Image, top_k: int = 10, top_labels: int = 5) -> dict:
        neighbors = self.query(image, top_k=top_k)
        valid = [n for n in neighbors if n.get("label") and n["label"] != "unknown"]
        if not valid:
            return {"label": "unknown", "predicted_category": "unknown",
                    "predicted_subcategory": None, "score": 0.0,
                    "candidate_labels": [], "neighbors": neighbors, "confidence": "low"}
        best = valid[0]
        return {
            "label": best["label"],
            "predicted_category": best["label"],
            "predicted_subcategory": best.get("subcategory"),
            "best_subcategory": best.get("subcategory"),
            "score": float(best.get("score", 0.0)),
            "candidate_labels": [n["label"] for n in valid[:top_labels]],
            "neighbors": neighbors,
            "confidence": "high",
        }

    def save_unknown_crop(self, crop: Image.Image, crop_id: int,
                          output_dir: str = "unknown_crops") -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"unknown_crop_{crop_id}_{int(time.time())}.png")
        crop.save(path)
        with open(os.path.join(output_dir, "unknown_crops.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"crop_id": crop_id, "crop_path": path,
                                 "timestamp": int(time.time())}) + "\n")
        return path


def load_swin_faiss_classifier(**kwargs) -> SwinFaissClassifier:
    return SwinFaissClassifier(**kwargs)
