# Payload: download the RPC (Retail Product Checkout) dataset and cache it in
# $BUCKET/rpc/. Sourced after _fetcher_lib.sh with __BUCKET__ substituted.
#
# RPC has no single stable mirror. We probe sources in order:
#   1) The diyer22/retail-product-checkout-dataset Kaggle mirror (global, stable).
#      Requires a Kaggle API token in $BUCKET/_secrets/kaggle.json (created by user
#      via Kaggle Account -> Create New API Token).
#   2) The raw GitHub release of RPC's published assets (smaller subset; checked
#      only if Kaggle path fails).
#
# If both fail, we exit with rc=2 and the trap still cleans up the VM.

BUCKET="__BUCKET__"
DEST="$BUCKET/rpc"
LOG=/tmp/fetch.log

_setup_logging "$LOG"
_setup_cleanup "$BUCKET" "$LOG" "rpc/fetch.log"

echo "=== fetch_rpc starting at $(date -u) ==="

apt-get update >/dev/null 2>&1 || true
apt-get install -y curl unzip python3 python3-pip >/dev/null 2>&1 || true

# Work on the boot disk, NOT tmpfs. The boot disk is 80 GB; RAM on the fetcher
# is small, so anything memory-backed will OOM on a 30 GB dataset.
DL=/var/tmp/dl
mkdir -p "$DL/raw" "$DL/extracted"
cd "$DL/raw"

# --- Source 1: Kaggle mirror via the kaggle CLI -----------------------------
echo "=== probing Kaggle source ==="
if gsutil -q stat "$BUCKET/_secrets/kaggle.json"; then
  echo "Kaggle token found in $BUCKET/_secrets/kaggle.json — using Kaggle CLI"
  pip install --quiet kaggle
  mkdir -p ~/.kaggle
  gsutil cp "$BUCKET/_secrets/kaggle.json" ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json

  # IMPORTANT: download WITHOUT --unzip. The --unzip flag streams decompression
  # through memory and OOM-killed the e2-small. We download the zip to disk,
  # upload the raw zip to GCS first (so the data is safe even if extraction
  # later fails), then unzip from disk to disk.
  echo "=== downloading RPC zip (no --unzip) ==="
  if kaggle datasets download -d diyer22/retail-product-checkout-dataset \
       --path "$DL/raw" 2>&1 | tee /tmp/kaggle.log; then
    echo "Kaggle download succeeded"
    ls -lh "$DL/raw"

    # Safety net: stash the raw archive(s) in the bucket before extracting.
    echo "=== uploading raw archive(s) -> $DEST/raw/ ==="
    gsutil -m cp "$DL/raw"/*.zip "$DEST/raw/" 2>/dev/null || true

    echo "=== extracting on disk ==="
    for z in "$DL/raw"/*.zip; do
      echo "unzip $z"
      unzip -q -o "$z" -d "$DL/extracted"
    done
  else
    echo "WARN: Kaggle download failed; trying secondary source"
    rm -rf "${DL:?}/raw"/* "${DL:?}/extracted"/*
  fi
else
  echo "no Kaggle token at $BUCKET/_secrets/kaggle.json — skipping Kaggle source"
fi

# --- Source 2: github release fallback (small subset, may not be full dataset) -
if [[ -z "$(ls -A "$DL/extracted" 2>/dev/null)" ]]; then
  echo "=== attempting github fallback ==="
  # The original RPC release page links to raw zips on the public github release
  # for github.com/RetailAI/RPC-Dataset (canonical author release).
  # If they ever 404, the script exits non-zero and the trap cleans up.
  GH_URL="https://github.com/RetailAI/RPC-Dataset/releases/download/v1.0/rpc.tar.gz"
  if curl --fail --silent --head "$GH_URL" >/dev/null; then
    echo "downloading $GH_URL"
    curl --fail --location --retry 5 --retry-delay 5 -o rpc.tar.gz "$GH_URL"
    tar -xzf rpc.tar.gz -C "$DL/extracted"
  else
    cat <<EOF
ERROR: no working RPC source.
  - Kaggle: token missing at $BUCKET/_secrets/kaggle.json or download failed.
  - GitHub: $GH_URL not reachable.

To enable Kaggle:
  1. Visit https://www.kaggle.com/settings/account -> Create New API Token
  2. gsutil cp ~/Downloads/kaggle.json $BUCKET/_secrets/kaggle.json
  3. Re-run: bash datasets/launch_dataset_fetcher.sh rpc
EOF
    exit 2
  fi
fi

echo "=== uploading raw -> $DEST/raw/ ==="
gsutil -m cp -r "$DL/raw/." "$DEST/raw/" 2>/dev/null || true

echo "=== uploading extracted -> $DEST/extracted/ ==="
gsutil -m cp -r "$DL/extracted/." "$DEST/extracted/"

echo "=== writing manifest ==="
gsutil ls -l -r "$DEST/**" > /tmp/manifest.txt
gsutil cp /tmp/manifest.txt "$DEST/manifest.txt"

echo "=== fetch_rpc done at $(date -u) ==="
