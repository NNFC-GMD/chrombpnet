#!/bin/bash
#SBATCH --job-name=chrombpnet-deepshap
#SBATCH --account=iu_0109
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/dcai/users/mateug/logs/chrombpnet.deepshap.%j.out
#SBATCH --error=/dcai/users/mateug/logs/chrombpnet.deepshap.%j.err

BASE=/dcai/users/mateug/chrombpnet_tutorial
IN=""
OUT=""
REGIONS=""
DEEPSHAP_OUT=""
N_REGIONS=30000
TMP=""
SYNC_INTERVAL_SECONDS=900

usage() {
  cat <<'EOF'
Usage: sbatch chrombpnet_deepshap_legacy.sbatch [options]

Options:
  --base PATH           Tutorial/project base directory.
  --in PATH             Input downloads directory with hg38 files.
  --out PATH            ChromBPNet model output directory.
  --regions BED         BED regions to interpret.
  --deepshap-out PATH   Persistent DeepSHAP output directory.
  --n-regions N         Number of regions to sample. Use 0 for all regions.
  --tmp PATH            Node-local temp directory.
  --sync-interval SEC   Periodic sync interval.
  -h, --help            Show this message.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --in) IN="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --regions) REGIONS="$2"; shift 2 ;;
    --deepshap-out) DEEPSHAP_OUT="$2"; shift 2 ;;
    --n-regions) N_REGIONS="$2"; shift 2 ;;
    --tmp) TMP="$2"; shift 2 ;;
    --sync-interval) SYNC_INTERVAL_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source /dcai/software/tenant/apps/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
conda activate /dcai/users/mateug/envs/chrombpnet-shap

# kundajelab-shap expects TensorFlow graph tensors and TF1-style session APIs.
export TF_USE_LEGACY_KERAS=1

export LD_LIBRARY_PATH=$(
  find "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia -type d -name lib 2>/dev/null \
    | paste -sd:
)${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export TF_FORCE_GPU_ALLOW_GROWTH=true

set -euo pipefail

mkdir -p /dcai/users/mateug/logs

IN=${IN:-$BASE/data/downloads}
OUT=${OUT:-$BASE/chrombpnet_model}
REGIONS=${REGIONS:-$OUT/auxiliary/filtered.peaks.bed}
DEEPSHAP_OUT=${DEEPSHAP_OUT:-$OUT/deepshap}
TMP=${TMP:-/tmp/mateug/${SLURM_JOB_ID:-manual}}
SYNC_PID=""

sync_outputs() {
  local src="$TMP/output/deepshap/"
  if [ -d "$src" ]; then
    echo "Checkpoint sync: $(date)"
    mkdir -p "$DEEPSHAP_OUT"
    rsync -a --partial --delay-updates "$src" "$DEEPSHAP_OUT/" || true
  fi
}

periodic_sync() {
  while true; do
    sleep "$SYNC_INTERVAL_SECONDS"
    sync_outputs
  done
}

cleanup() {
  local status=$?
  if [ -n "${SYNC_PID:-}" ]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
  sync_outputs
  if [ "$status" -eq 0 ]; then
    echo "Final result sync: $(date)"
    mkdir -p "$DEEPSHAP_OUT"
    rsync -a --partial --delay-updates "$TMP/output/deepshap/" "$DEEPSHAP_OUT/"
    echo "End: $(date)"
    rm -rf "$TMP"
  else
    echo "Job exited with status $status at $(date)"
    echo "Leaving temp dir for debugging/rescue: $TMP"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'echo "Received termination signal at $(date)"; exit 143' TERM INT

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

python - <<'PY'
import numpy as np
import tensorflow as tf
print("NumPy", np.__version__)
print("TensorFlow", tf.__version__)
print("GPU devices", tf.config.list_physical_devices("GPU"))
import IPython
print("IPython", IPython.__version__)
import shap
print("SHAP import OK", getattr(shap, "__version__", "unknown"))
PY

rm -rf "$TMP"
mkdir -p "$TMP"/{downloads,data,models,output/deepshap}

echo "Copying inputs: $(date)"
cp "$IN/hg38.fa" "$TMP/downloads/"
cp "$IN/hg38.chrom.sizes" "$TMP/downloads/"
cp "$IN/hg38.fa.fai" "$TMP/downloads/" 2>/dev/null || true
cp "$OUT/models/chrombpnet_nobias.h5" "$TMP/models/"

if [ "$N_REGIONS" -gt 0 ]; then
  shuf --random-source=<(yes 42) -n "$N_REGIONS" "$REGIONS" > "$TMP/data/interpret_regions.bed"
else
  cp "$REGIONS" "$TMP/data/interpret_regions.bed"
fi
echo "Input copy done: $(date)"

periodic_sync &
SYNC_PID=$!

chrombpnet contribs_bw \
  -m "$TMP/models/chrombpnet_nobias.h5" \
  -r "$TMP/data/interpret_regions.bed" \
  -g "$TMP/downloads/hg38.fa" \
  -c "$TMP/downloads/hg38.chrom.sizes" \
  -op "$TMP/output/deepshap/chrombpnet_nobias" \
  -pc profile counts \
  -t 1
