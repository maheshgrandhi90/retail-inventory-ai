# Module 2b — SWIN + FAISS retrieval classification (+ analytics/BI dashboard)

An **alternative to the trained classifier**. Instead of a softmax head over thousands of
product classes (which collapses on rare classes — the near-zero result the team hit), this
**embeds each product crop with a SWIN transformer and nearest-neighbour-searches a FAISS index
of ~692k pre-embedded, labeled crops**. The nearest neighbour's category/subcategory is the
prediction. Retrieval degrades gracefully across many rare classes, so it beats the classifiers
on this data.

Origin: adapted from the teammate's repo
[`Richa-max/dense-shelf-images-object-detection`](https://github.com/Richa-max/dense-shelf-images-object-detection).
Only the pieces required for inference were brought over; the code was cleaned to use explicit,
configurable asset paths (no CWD `*.csv` globbing) and our **own** YOLO detector.

## Pipeline

```
shelf image ──YOLO(detection/artifacts/v11/best.pt)──▶ crops ──▶ SWIN embed ──▶ FAISS search
                                                                       │
                                          nearest neighbour's (category, subcategory)
                                                                       │
                                       persist to inventory (SQLite) ──▶ analytics + NL BI
```

## Files

| File | Role |
|---|---|
| `swin_faiss.py` | SWIN encoder + FAISS index; `classify(crop)` → category/subcategory + neighbours |
| `pipeline.py` | `analyze_image()`: YOLO detect → crop → classify → annotated image + metrics |
| `assets/` | model + index + label maps (large files gitignored — see below) |
| `../backend/inventory_db.py` | SQLite persistence of scans/items (Module 4) |
| `../bi_interface/bi_engine.py` | natural-language Q&A over the inventory (Module 5) |
| `../frontend/app.py` | Streamlit analytics + BI dashboard (Module 7) |

## Assets (provisioning)

The large binaries are **gitignored** (like datasets/model checkpoints elsewhere in this repo).
Only the small model-config JSONs are committed, to document the exact encoder/preprocessor.

Required layout under `retrieval/assets/`:

```
assets/
├── swin_model_assets/
│   ├── config.json            ← committed
│   └── model.safetensors      ← ~110 MB, gitignored
├── swin_processor_assets/
│   └── preprocessor_config.json   ← committed
├── swin_faiss_index.bin       ← ~2.1 GB, gitignored (the FAISS query target)
├── swin_faiss_indexed_image_paths.csv   ← ~16 MB, gitignored (index row → image_path)
└── labels/
    └── train_product_category_58.csv    ← ~80 MB, gitignored (image_path → category/subcategory)
```

Provision them from the teammate repo (Git LFS):

```bash
git clone https://github.com/Richa-max/dense-shelf-images-object-detection.git
cd dense-shelf-images-object-detection && git lfs install --local && git lfs pull
# then copy into this repo (swin_full_embeddings.npy is NOT needed — index-build only):
DST=/path/to/retail-inventory-ai/retrieval/assets
cp -r swin_model_assets/* "$DST/swin_model_assets/"
cp -r swin_processor_assets/* "$DST/swin_processor_assets/"
cp swin_faiss_index.bin swin_faiss_indexed_image_paths.csv "$DST/"
cp train_product_category_58.csv "$DST/labels/"
```

> `swin_faiss_index.bin` is a **flat** index (~2.1 GB, brute-force). It works but doesn't scale
> well past ~700k vectors; migrate to an IVF/HNSW index for larger sets.

## Run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[retrieval]"
KMP_DUPLICATE_LIB_OK=TRUE streamlit run frontend/app.py   # http://localhost:8501
```

`KMP_DUPLICATE_LIB_OK=TRUE` is required on macOS (torch and faiss each bundle libomp); the code
also sets it defensively at import.

## Label space note

This module uses the teammate's **58-category** taxonomy (`train_product_category_58.csv`),
which is finer-grained than this repo's own **18/48** normalized taxonomy
(`classification/taxonomy.py`). Reconciling the two is a tracked follow-up.
