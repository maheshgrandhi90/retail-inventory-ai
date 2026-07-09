# Adding Mahesh's SKU OCR Setup To The Teammate Repo

Use this guide when working inside the teammate repo:

```text
dense-shelf-images-object-detection/
```

The goal is to create a self-contained folder:

```text
dense-shelf-images-object-detection/
└── mgrandhi/
```

This folder should contain only Mahesh's code, docs, database schema, SKU/OCR benchmark harness,
and Streamlit UI wrapper. Do not copy large LFS/model assets into `mgrandhi/`; use the teammate
repo's existing YOLO, SWIN, FAISS, `.pt`, and asset files from their current locations.

## 1. Start From The Teammate Repo

```bash
cd ~/Projects/dense-shelf-images-object-detection
git checkout main
git pull
git checkout -b mgrandhi-sku-ocr-setup
```

## 2. Create The Folder Layout

```bash
mkdir -p mgrandhi/backend mgrandhi/autolabel mgrandhi/docs mgrandhi/config
touch mgrandhi/backend/__init__.py
touch mgrandhi/autolabel/__init__.py
```

Recommended final structure:

```text
mgrandhi/
├── README.md
├── requirements.txt
├── .env.example
├── app.py
├── backend/
│   ├── __init__.py
│   ├── inventory_db.py
│   ├── init_schema.py
│   └── schema.sql
├── autolabel/
│   ├── __init__.py
│   ├── sku_vlm.py
│   ├── launch_sku_vlm_benchmark.sh
│   ├── sku_vlm_benchmark.sh
│   ├── launch_open_vlm_endpoint.sh
│   └── open_vlm_endpoint_startup.sh
├── docs/
│   ├── database_schema.md
│   └── open_vlm_sku_benchmark.md
└── config/
    └── paths.py
```

## 3. Copy Required Files From This Repo

Assuming both repos are under `~/Projects`:

```bash
cd ~/Projects/dense-shelf-images-object-detection

cp ../retail-inventory-ai/backend/inventory_db.py mgrandhi/backend/
cp ../retail-inventory-ai/backend/init_schema.py mgrandhi/backend/
cp ../retail-inventory-ai/backend/schema.sql mgrandhi/backend/

cp ../retail-inventory-ai/autolabel/sku_vlm.py mgrandhi/autolabel/
cp ../retail-inventory-ai/autolabel/sku_vlm_benchmark.sh mgrandhi/autolabel/
cp ../retail-inventory-ai/autolabel/launch_sku_vlm_benchmark.sh mgrandhi/autolabel/
cp ../retail-inventory-ai/autolabel/open_vlm_endpoint_startup.sh mgrandhi/autolabel/
cp ../retail-inventory-ai/autolabel/launch_open_vlm_endpoint.sh mgrandhi/autolabel/

cp ../retail-inventory-ai/docs/database_schema.md mgrandhi/docs/
cp ../retail-inventory-ai/docs/open_vlm_sku_benchmark.md mgrandhi/docs/

cp ../retail-inventory-ai/frontend/app.py mgrandhi/app.py
```

## 4. Add A Path Config For Teammate Assets

Create `mgrandhi/config/paths.py`:

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MGRANDHI_ROOT = Path(__file__).resolve().parents[1]

# Adjust these paths to match the teammate repo's actual asset layout.
YOLO_WEIGHTS = REPO_ROOT / "best.pt"
FAISS_INDEX = REPO_ROOT / "assets" / "swin_faiss_index.bin"
SWIN_MODEL_DIR = REPO_ROOT / "assets" / "swin_model_assets"
SWIN_PROCESSOR_DIR = REPO_ROOT / "assets" / "swin_processor_assets"
LABELS_DIR = REPO_ROOT / "assets" / "labels"

INVENTORY_DB = MGRANDHI_ROOT / "inventory.db"
```

If the teammate repo stores these assets elsewhere, only update this file. Do not move or duplicate
large files.

## 5. Add Requirements

Create `mgrandhi/requirements.txt`:

```text
numpy>=1.26,<2.0
pandas>=2.2
pillow>=10.2
opencv-python>=4.9
torch>=2.2
torchvision>=0.17
transformers>=4.40
faiss-cpu>=1.8
ultralytics>=8.2
streamlit>=1.33
plotly>=5.20
requests>=2.31
google-genai>=0.3
```

Add extra dependencies only if the teammate dashboard or SWIN/FAISS code requires them.

## 6. Add A Local README

Create `mgrandhi/README.md`:

```markdown
# Mahesh SKU OCR + Inventory Demo

This folder contains Mahesh's SKU/OCR, database schema, benchmark harness, and Streamlit UI setup.

It intentionally does not include large model files. It uses the parent teammate repo's existing
LFS assets for YOLO, SWIN, FAISS, and labels.

## Setup

\`\`\`bash
cd ~/Projects/dense-shelf-images-object-detection
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r mgrandhi/requirements.txt
\`\`\`

## Run

\`\`\`bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 streamlit run mgrandhi/app.py
\`\`\`

## Notes

- Update `mgrandhi/config/paths.py` if asset paths differ.
- Do not copy `.pt`, FAISS, SWIN, or other large LFS assets into this folder.
- Start SKU/OCR extraction with a small crop limit, for example `3`.
```

## 7. Update Imports And Asset Paths

The copied `mgrandhi/app.py` currently comes from this repo and may import:

```python
from backend import inventory_db as db
from bi_interface import bi_engine
from retrieval import pipeline
```

Inside the teammate repo, update these imports so they either:

- use copied `mgrandhi` modules, or
- call the teammate repo's existing detection/SWIN/FAISS functions directly.

Preferred clean approach:

```python
from mgrandhi.backend import inventory_db as db
```

For retrieval/classification, either copy our small retrieval wrapper into `mgrandhi/retrieval/` and
point it to teammate assets through `mgrandhi/config/paths.py`, or adapt `mgrandhi/app.py` to call
the teammate repo's current pipeline.

## 8. Run Checks

```bash
cd ~/Projects/dense-shelf-images-object-detection
source .venv/bin/activate
python -m compileall mgrandhi
```

Then run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 streamlit run mgrandhi/app.py
```

## 9. Commit In Teammate Repo

```bash
git status
git add mgrandhi
git commit -m "Add Mahesh SKU OCR and inventory demo setup"
git push -u origin mgrandhi-sku-ocr-setup
```

## Cost Reminder

Managed Model Garden endpoints cost money while deployed with active replicas. For demo/testing:

1. Deploy PaliGemma only when needed.
2. Test with a small crop limit first.
3. Undeploy/delete the endpoint after the session.
4. Keep large self-hosted GPU VMs deleted unless actively benchmarking.
