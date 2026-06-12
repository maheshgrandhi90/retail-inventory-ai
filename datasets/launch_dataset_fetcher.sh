#!/bin/bash
# Launch a no-public-IP e2-small VM that downloads one dataset into the
# datasets bucket and self-destructs.
#
#   export PROJECT_ID=<your-gcp-project>
#   bash datasets/launch_dataset_fetcher.sh sku110k         # or coco2017, rpc, ...
#
# The dataset name maps to datasets/fetch_<name>.sh.
set -euo pipefail

DATASET="${1:?usage: $0 <dataset>   (one of: sku110k, coco2017, rpc, ...)}"

: "${PROJECT_ID:?set PROJECT_ID (e.g. export PROJECT_ID=ehc-mgrandhi-bc801a)}"
ZONE="${ZONE:-us-central1-a}"
REGION="${ZONE%-*}"
DATASETS_BUCKET="${DATASETS_BUCKET:-gs://${PROJECT_ID}-datasets}"
INSTANCE="${INSTANCE:-dataset-fetcher-${DATASET}}"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

# Per-dataset machine type. Most fetchers are I/O-bound and fine on e2-small.
# COCO 2017 has 164k tiny files; the upload is CPU-bound serializing them, so it
# needs the parallelism of a bigger machine to finish in <30 min instead of ~5 h.
case "$DATASET" in
  coco2017) MACHINE_TYPE_DEFAULT=e2-standard-8 ;;   # 164k tiny files, upload CPU-bound
  rpc)      MACHINE_TYPE_DEFAULT=e2-standard-4 ;;    # ~30 GB zip; needs RAM for unzip (e2-small OOM'd)
  *)        MACHINE_TYPE_DEFAULT=e2-small ;;
esac
MACHINE_TYPE="${MACHINE_TYPE:-$MACHINE_TYPE_DEFAULT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$SCRIPT_DIR/fetch_${DATASET}.sh"
LIB="$SCRIPT_DIR/_fetcher_lib.sh"

if [[ ! -f "$PAYLOAD" ]]; then
  echo "ERROR: $PAYLOAD not found"
  echo "Available fetchers: $(ls "$SCRIPT_DIR"/fetch_*.sh 2>/dev/null | xargs -n1 basename | sed 's/fetch_//;s/.sh//' | tr '\n' ' ')"
  exit 1
fi

# --- Cloud NAT precheck (same as detection/launch_vm.sh) -----------------
ROUTER="$(gcloud compute routers list --regions="$REGION" --format='value(name)' 2>/dev/null | head -1)"
NAT_FOUND=""
if [[ -n "$ROUTER" ]]; then
  NAT_FOUND="$(gcloud compute routers describe "$ROUTER" --region="$REGION" --format='value(nats[].name)' 2>/dev/null)"
fi
if [[ -z "$NAT_FOUND" ]]; then
  cat >&2 <<EOF
ERROR: no Cloud NAT in region '$REGION'.
       The no-public-IP fetcher VM will hang on every download.

Enable Cloud NAT (one-time) with:

  gcloud compute routers create nat-router-$REGION \\
    --region=$REGION --network=default
  gcloud compute routers nats create nat-config \\
    --router=nat-router-$REGION --region=$REGION \\
    --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges
EOF
  exit 1
fi

# --- Verify the bucket exists --------------------------------------------
if ! gcloud storage buckets describe "$DATASETS_BUCKET" >/dev/null 2>&1; then
  echo "ERROR: $DATASETS_BUCKET does not exist. Run datasets/setup_bucket.sh first."
  exit 1
fi

# --- Render the startup script: lib + payload + __BUCKET__ substitution ---
STARTUP="$(mktemp)"
{
  cat "$LIB"
  echo
  cat "$PAYLOAD"
} | sed -e "s|__BUCKET__|${DATASETS_BUCKET}|g" > "$STARTUP"

cat <<EOF
Project        : $PROJECT_ID
Zone / Region  : $ZONE / $REGION
Datasets bucket: $DATASETS_BUCKET
Instance       : $INSTANCE
Dataset        : $DATASET
Machine type   : $MACHINE_TYPE
Service account: $SERVICE_ACCOUNT
External IP    : NONE  (--no-address; egress via Cloud NAT)
EOF
echo "Creating fetcher VM ($MACHINE_TYPE, ubuntu-2204-lts)..."

gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --subnet=default \
  --no-address \
  --service-account="$SERVICE_ACCOUNT" \
  --machine-type="$MACHINE_TYPE" \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=80GB \
  --boot-disk-type=pd-balanced \
  --metadata-from-file=startup-script="$STARTUP" \
  --scopes=https://www.googleapis.com/auth/cloud-platform

rm -f "$STARTUP"

cat <<EOF

VM '$INSTANCE' is booting. The fetcher self-deletes when done.

Monitor (IAP tunnel — direct SSH is blocked because the VM has no external IP):
  gcloud compute ssh $INSTANCE --zone $ZONE --tunnel-through-iap \\
    --command "sudo tail -f /tmp/fetch.log"

When the instance disappears from 'gcloud compute instances list', the cache is ready:
  gsutil ls -l $DATASETS_BUCKET/$DATASET/
  gsutil cat $DATASETS_BUCKET/$DATASET/fetch.log | tail -20
EOF
