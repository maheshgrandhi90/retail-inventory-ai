# Retail Inventory AI

**Intelligent Retail Product Detection & Inventory Management System**
Computer Vision (YOLOv8 + ResNet-50 + CLIP + LLaVA) + LLM-based Business Intelligence.

An 8-week / 2-month project that automates product detection from shelf images, maintains a
live inventory database, and exposes a natural-language business-intelligence interface for
non-technical users. Full spec: `~/Downloads/Project_Proposal_Retail.pdf`.

---

## 🔄 How to "resume"

This project is built to be picked up across many sessions. To continue work:

1. Open this folder in Claude Code.
2. Say **"resume"**.
3. Claude reads [`PROJECT_MEMORY.md`](PROJECT_MEMORY.md) (the single source of truth for current
   status and next step) and continues from there. A persistent Claude memory also points back to
   this file so resume works even from a fresh context.

Every working session ends by updating `PROJECT_MEMORY.md` and appending to the running logs
(`DECISIONS.md`, `ISSUES.md`, `RESULTS.md`). **These logs are the raw material for the final
LaTeX report** — we assemble the report *from* them, not from memory.

---

## Project layout

| Path | Module | Week | Status |
|---|---|---|---|
| `detection/` | M1 — YOLOv8 detection (SKU-110K) | 3 | ✅ done (mAP@0.5≈0.92) |
| `classification/` | M2 — confidence-gated cat/subcat classifier (SKU-110K crops) | 4 | ✅ trained (see RESULTS.md) |
| `retrieval/` | M2b — SWIN + FAISS retrieval classifier (alternative to the trained classifier) | 4 | ✅ integrated + running |
| `autolabel/` | M2/M3 — CLIP + GCP VLM (Vertex Gemini / Gemma) labeling & fallback | 4–5 | 🚧 scaffolded |
| `backend/` | M4 — inventory store (SQLite) + (future) FastAPI | 5 | 🚧 SQLite store live |
| `bi_interface/` | M5 — natural-language BI (rule-based now, Ollama-ready) | 6 | 🚧 v1 live |
| `frontend/` | M7 — Streamlit analytics + BI dashboard | 7 | 🚧 v1 live |
| `notebooks/` | Demo / checkpoint notebooks | — | 🚧 |
| `report/` | LaTeX final report | 8 | 🚧 skeleton |
| `docs/` | Diagrams, literature summaries | 1 | ⬜ |

Tracking files: `PROJECT_MEMORY.md` (resume), `DECISIONS.md`, `ISSUES.md`, `RESULTS.md`.

---

## Setup

Local Python is 3.14 (too new for the ML stack). Use a **Python 3.11** venv:

```bash
cd retail-inventory-ai
python3.11 -m venv .venv
source .venv/bin/activate

# Install only what you need per module, e.g. detection:
pip install -e ".[detection,dev]"
# or everything:
pip install -e ".[all]"

cp .env.example .env   # then fill in PROJECT_ID, BUCKET, etc.
```

> Heavy training runs on **GCP** (L4 GPU) or **Colab** — not the local macOS/CPU machine.

---

## 🚀 Run the analytics + BI dashboard (Modules 2b / 4 / 5 / 7)

An end-to-end shelf app that runs **locally**: upload a shelf image → YOLO detects products →
SWIN + FAISS retrieval classifies each crop → the dashboard shows KPIs, interactive analytics
charts, and a natural-language Business-Intelligence panel over an accumulating SQLite inventory.

```bash
cd retail-inventory-ai
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[retrieval]"

# One-time: provision the SWIN/FAISS assets (~2.3 GB, gitignored).
# See retrieval/README.md for the exact copy commands from the teammate repo (Git LFS).

# Launch (KMP flag required on macOS — torch + faiss both bundle libomp):
KMP_DUPLICATE_LIB_OK=TRUE streamlit run frontend/app.py   # -> http://localhost:8501
```

Tabs: **Detection** (annotated image + KPIs + CSV export) · **Analytics** (category bar/donut,
subcategory treemap, empty-space gauge) · **Business Intelligence** (NL Q&A; auto-uses Ollama if
running, else a deterministic rule-based engine) · **Inventory History** (trends across scans).

The retrieval classifier reuses **our own** detector (`detection/artifacts/v11/best.pt`). Full
details + asset provisioning in [`retrieval/README.md`](retrieval/README.md).

---

## Today's deliverables (Week 3 — Detection checkpoint)

1. **Train YOLOv8m on SKU-110K** on a GCP L4 GPU VM (10h cap, auto-teardown, results → GCS).
   See [`detection/README.md`](detection/README.md).
2. **Before/after checkpoint notebook** — [`notebooks/01_detection_checkpoint.ipynb`](notebooks/01_detection_checkpoint.ipynb):
   YOLOv8m COCO weights *before* training vs our fine-tuned `best.pt` *after*, on the same shelf
   images. Upload to Colab and "Run all."

**Evaluation metrics captured:** mAP@0.5 and mAP@0.5:0.95 (targets: mAP@0.5 ≥ 0.70).
