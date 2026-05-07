#!/bin/bash
#SBATCH --job-name=chrombpnet-modisco
#SBATCH --account=iu_0109
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/dcai/users/mateug/logs/chrombpnet.modisco.%j.out
#SBATCH --error=/dcai/users/mateug/logs/chrombpnet.modisco.%j.err

BASE=/dcai/users/mateug/chrombpnet_tutorial
OUT=""
SCORES_PREFIX=""
MODISCO_OUT=""
MAX_SEQLETS=50000
WINDOW=500
RUN_PROFILE=1
RUN_COUNTS=1
MOTIF_MEME=""
TMP=""

usage() {
  cat <<'EOF'
Usage: sbatch chrombpnet_modisco.sbatch [options]

Options:
  --base PATH           Tutorial/project base directory.
  --out PATH            ChromBPNet model output directory.
  --scores-prefix PATH  Prefix for DeepSHAP score files.
  --modisco-out PATH    Persistent MoDISco output directory.
  --max-seqlets N       Maximum seqlets for modisco motifs.
  --window N            Window size for modisco motifs.
  --run-profile 0|1     Run profile MoDISco.
  --run-counts 0|1      Run counts MoDISco.
  --motif-meme PATH     Motif MEME database for reports.
  --tmp PATH            Node-local temp directory.
  -h, --help            Show this message.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --scores-prefix) SCORES_PREFIX="$2"; shift 2 ;;
    --modisco-out) MODISCO_OUT="$2"; shift 2 ;;
    --max-seqlets) MAX_SEQLETS="$2"; shift 2 ;;
    --window) WINDOW="$2"; shift 2 ;;
    --run-profile) RUN_PROFILE="$2"; shift 2 ;;
    --run-counts) RUN_COUNTS="$2"; shift 2 ;;
    --motif-meme) MOTIF_MEME="$2"; shift 2 ;;
    --tmp) TMP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source /dcai/software/tenant/apps/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
conda activate /dcai/users/mateug/envs/chrombpnet-h100

set -euo pipefail

mkdir -p /dcai/users/mateug/logs

OUT=${OUT:-$BASE/chrombpnet_model}
SCORES_PREFIX=${SCORES_PREFIX:-$OUT/deepshap/chrombpnet_nobias}
MODISCO_OUT=${MODISCO_OUT:-$OUT/modisco}
MOTIF_MEME=${MOTIF_MEME:-$(print_meme_motif_file)}
TMP=${TMP:-/tmp/mateug/${SLURM_JOB_ID:-manual}}

cleanup() {
  local status=$?
  if [ "$status" -eq 0 ]; then
    echo "Final result sync: $(date)"
    mkdir -p "$MODISCO_OUT"
    rsync -a --partial --delay-updates "$TMP/output/modisco/" "$MODISCO_OUT/"
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
echo "Scores prefix: $SCORES_PREFIX"
echo "Motif database: $MOTIF_MEME"

modisco motifs --help >/dev/null
modisco report --help >/dev/null

rm -rf "$TMP"
mkdir -p "$TMP"/{scores,output/modisco}

cp "$SCORES_PREFIX.profile_scores.h5" "$TMP/scores/profile_scores.h5"
if [ "$RUN_COUNTS" -eq 1 ]; then
  cp "$SCORES_PREFIX.counts_scores.h5" "$TMP/scores/counts_scores.h5"
fi

if [ "$RUN_PROFILE" -eq 1 ]; then
  modisco motifs \
    -i "$TMP/scores/profile_scores.h5" \
    -n "$MAX_SEQLETS" \
    -o "$TMP/output/modisco/modisco_results_profile_scores.h5" \
    -w "$WINDOW"

  modisco report \
    -i "$TMP/output/modisco/modisco_results_profile_scores.h5" \
    -o "$TMP/output/modisco/modisco_profile" \
    -m "$MOTIF_MEME"
fi

if [ "$RUN_COUNTS" -eq 1 ]; then
  modisco motifs \
    -i "$TMP/scores/counts_scores.h5" \
    -n "$MAX_SEQLETS" \
    -o "$TMP/output/modisco/modisco_results_counts_scores.h5" \
    -w "$WINDOW"

  modisco report \
    -i "$TMP/output/modisco/modisco_results_counts_scores.h5" \
    -o "$TMP/output/modisco/modisco_counts" \
    -m "$MOTIF_MEME"
fi
