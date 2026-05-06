#!/bin/bash
#SBATCH --job-name=chrombpnet-pred-bw
#SBATCH --account=iu_0109
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/dcai/users/mateug/logs/chrombpnet.pred_bw.%j.out
#SBATCH --error=/dcai/users/mateug/logs/chrombpnet.pred_bw.%j.err

BASE=/dcai/users/mateug/chrombpnet_tutorial
IN=""
OUT=""
REGIONS=""
OBSERVED_BW=""
USE_OBSERVED=1
PRED_OUT=""
TRACK_PREFIX=head
BATCH_SIZE=64
TMP=""

usage() {
  cat <<'EOF'
Usage: sbatch --export=NIL chrombpnet_pred_bw_h100.sbatch [options]

Options:
  --base PATH           Tutorial/project base directory.
  --in PATH             Input downloads directory with hg38 files.
  --out PATH            ChromBPNet model output directory.
  --regions BED         BED regions for prediction.
  --observed-bw BW      Observed head/count bigWig for metrics.
  --no-observed-bw      Skip observed-track metrics.
  --pred-out PATH       Persistent prediction output directory.
  --track-prefix NAME   Prefix for generated prediction tracks.
  --batch-size N        Prediction batch size.
  --tmp PATH            Node-local temp directory.
  -h, --help            Show this message.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --in) IN="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --regions) REGIONS="$2"; shift 2 ;;
    --observed-bw) OBSERVED_BW="$2"; USE_OBSERVED=1; shift 2 ;;
    --no-observed-bw) USE_OBSERVED=0; shift ;;
    --pred-out) PRED_OUT="$2"; shift 2 ;;
    --track-prefix) TRACK_PREFIX="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --tmp) TMP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source /dcai/software/tenant/apps/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
conda activate /dcai/users/mateug/envs/chrombpnet-h100

export LD_LIBRARY_PATH=$(
  find "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia -type d -name lib 2>/dev/null \
    | paste -sd:
)${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export TF_FORCE_GPU_ALLOW_GROWTH=true

set -euo pipefail

mkdir -p /dcai/users/mateug/logs

export CUDA_CACHE_PATH=/dcai/users/mateug/.cache/cuda
export CUDA_CACHE_MAXSIZE=2147483648
mkdir -p "$CUDA_CACHE_PATH"

IN=${IN:-$BASE/data/downloads}
OUT=${OUT:-$BASE/chrombpnet_model}
REGIONS=${REGIONS:-$OUT/auxiliary/filtered.peaks.bed}
OBSERVED_BW=${OBSERVED_BW:-$OUT/auxiliary/data_unstranded.bw}
PRED_OUT=${PRED_OUT:-$OUT/pred_bw}
TMP=${TMP:-/tmp/mateug/${SLURM_JOB_ID:-manual}}

cleanup() {
  local status=$?
  if [ "$status" -eq 0 ]; then
    echo "Final result sync: $(date)"
    mkdir -p "$PRED_OUT"
    rsync -a --partial --delay-updates "$TMP/output/" "$PRED_OUT/"
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
nvidia-smi
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

rm -rf "$TMP"
mkdir -p "$TMP"/{downloads,data,models,output}

echo "Copying inputs: $(date)"
cp "$IN/hg38.fa" "$TMP/downloads/"
cp "$IN/hg38.chrom.sizes" "$TMP/downloads/"
cp "$IN/hg38.fa.fai" "$TMP/downloads/" 2>/dev/null || true
cp "$REGIONS" "$TMP/data/regions.bed"
cp "$OUT/models/chrombpnet.h5" "$TMP/models/"
cp "$OUT/models/chrombpnet_nobias.h5" "$TMP/models/"

observed_arg=()
if [ "$USE_OBSERVED" -eq 1 ]; then
  cp "$OBSERVED_BW" "$TMP/data/observed_head.bw"
  observed_arg=(-bw "$TMP/data/observed_head.bw")
fi
echo "Input copy done: $(date)"

chrombpnet pred_bw \
  -cmb "$TMP/models/chrombpnet_nobias.h5" \
  -r "$TMP/data/regions.bed" \
  -g "$TMP/downloads/hg38.fa" \
  -c "$TMP/downloads/hg38.chrom.sizes" \
  -op "$TMP/output/$TRACK_PREFIX" \
  -bs "$BATCH_SIZE" \
  -t 1 \
  "${observed_arg[@]}"

chrombpnet pred_bw \
  -cm "$TMP/models/chrombpnet.h5" \
  -r "$TMP/data/regions.bed" \
  -g "$TMP/downloads/hg38.fa" \
  -c "$TMP/downloads/hg38.chrom.sizes" \
  -op "$TMP/output/$TRACK_PREFIX" \
  -bs "$BATCH_SIZE" \
  -t 1 \
  "${observed_arg[@]}"
