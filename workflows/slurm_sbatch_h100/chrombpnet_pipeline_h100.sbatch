#!/bin/bash
#SBATCH --job-name=chrombpnet
#SBATCH --account=iu_0109
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/dcai/users/mateug/logs/chrombpnet.%j.out
#SBATCH --error=/dcai/users/mateug/logs/chrombpnet.%j.err

source /dcai/software/tenant/apps/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
conda activate /dcai/users/mateug/envs/chrombpnet-h100

export LD_LIBRARY_PATH=$(
  find "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia -type d -name lib 2>/dev/null \
    | paste -sd:
)${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export TF_FORCE_GPU_ALLOW_GROWTH=true

# GPU sanity check
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

set -euo pipefail

mkdir -p /dcai/users/mateug/logs

export CUDA_CACHE_PATH=/dcai/users/mateug/.cache/cuda
export CUDA_CACHE_MAXSIZE=2147483648
mkdir -p "$CUDA_CACHE_PATH"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

nvidia-smi

BASE=/dcai/users/mateug/chrombpnet_tutorial
IN=$BASE/data/downloads
DATA=$BASE/data
BIAS=$BASE/bias_model
OUT=$BASE/chrombpnet_model
TMP=/tmp/$USER/$SLURM_JOB_ID
SYNC_INTERVAL_SECONDS=600
SYNC_PID=""

sync_checkpoints() {
  local src="$TMP/output/chrombpnet_model/"

  if [ -d "$src" ]; then
    echo "Checkpoint sync: $(date)"
    mkdir -p "$OUT"

    rsync -a --partial --delay-updates --prune-empty-dirs \
      --include='*/' \
      --include='models/***' \
      --include='logs/***' \
      --include='*.args.json' \
      --include='*.params.json' \
      --exclude='*' \
      "$src" "$OUT/" || true
  fi
}

periodic_sync() {
  while true; do
    sleep "$SYNC_INTERVAL_SECONDS"
    sync_checkpoints
  done
}

cleanup() {
  local status=$?

  if [ -n "${SYNC_PID:-}" ]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi

  sync_checkpoints

  if [ "$status" -eq 0 ]; then
    echo "Final result sync: $(date)"
    mkdir -p "$OUT"
    rsync -a --partial --delay-updates "$TMP/output/chrombpnet_model/" "$OUT/"
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

mkdir -p "$TMP"
rm -rf "$TMP/output/chrombpnet_model"
mkdir -p "$TMP"/{downloads,data,splits,bias_model,output}

echo "Using temp dir: $TMP"

periodic_sync &
SYNC_PID=$!

echo "Copying inputs: $(date)"
cp "$IN/rep1_fragments.bed" "$TMP/downloads/"
cp "$IN/hg38.fa" "$TMP/downloads/"
cp "$IN/hg38.chrom.sizes" "$TMP/downloads/"
cp "$DATA/peaks_no_blacklist.bed" "$TMP/data/"
cp "$DATA/output_negatives.bed" "$TMP/data/"
cp "$DATA/splits/fold_0.json" "$TMP/splits/"
cp "$BIAS/ENCSR868FGK_bias_fold_0.h5" "$TMP/bias_model/"
cp "$IN/hg38.fa.fai" "$TMP/downloads/" 2>/dev/null || true
echo "Input copy done: $(date)"

chrombpnet pipeline \
  -ifrag "$TMP/downloads/rep1_fragments.bed" \
  -d "ATAC" \
  -g "$TMP/downloads/hg38.fa" \
  -c "$TMP/downloads/hg38.chrom.sizes" \
  -p "$TMP/data/peaks_no_blacklist.bed" \
  -n "$TMP/data/output_negatives.bed" \
  -fl "$TMP/splits/fold_0.json" \
  -b "$TMP/bias_model/ENCSR868FGK_bias_fold_0.h5" \
  -o "$TMP/output/chrombpnet_model" \
  --skip-shap
