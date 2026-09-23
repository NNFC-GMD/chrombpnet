#!/usr/bin/env bash
# Produce legacy parity goldens on the molab box, CPU only, inside the user's legacy chrombpnet container.
# Read-only on the container and the data tree; writes only to $OUT.
#   molab sbatch -J cbp_goldens -D /marimo/chrombpnet/.scratch -c 2 -t 60 --wrap 'bash /marimo/chrombpnet/tests/goldens/run_goldens.sh'
set -euo pipefail

SANDBOX=${SANDBOX:-/marimo/containers/chrombpnet_sandbox}
R=${RESULTS:-/marimo/data/test_data_d0/results}
GENOME=${GENOME:-/marimo/data/references/hg38/Sequence/IGVFFI0653VCGH.fasta}
OUT=${OUT:-/marimo/chrombpnet/.scratch/goldens}
SCRIPT=$(cd "$(dirname "$0")" && pwd)/make_goldens.py
BM=$R/bias_models
FM=$R/full_models/d0_all_fold_0
NT=${SLURM_CPUS_PER_TASK:-2}
mkdir -p "$OUT" "$OUT/.mpl"

export APPTAINERENV_CUDA_VISIBLE_DEVICES=""
export APPTAINERENV_TF_CPP_MIN_LOG_LEVEL=2
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_MPLCONFIGDIR=$OUT/.mpl
export APPTAINERENV_OMP_NUM_THREADS=$NT
export APPTAINERENV_TF_NUM_INTRAOP_THREADS=$NT
export APPTAINERENV_TF_NUM_INTEROP_THREADS=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

legacy() { apptainer exec --cleanenv --bind /marimo:/marimo "$SANDBOX" python3 "$SCRIPT" "$@"; }

legacy_versions() {
    apptainer exec --cleanenv --bind /marimo:/marimo "$SANDBOX" python3 -c '
import json, sys, numpy, tensorflow as tf, shap, chrombpnet
print(json.dumps({"python": sys.version.split()[0], "tensorflow": tf.__version__, "numpy": numpy.__version__,
                  "shap": getattr(shap, "__version__", "?"), "chrombpnet": chrombpnet.__file__}))'
}

START=$(date -u +%FT%TZ)
legacy_versions > "$OUT/versions.json"
cat "$OUT/versions.json"

bias_models=()
for b in 03 03_bs128 05 055 065; do
    f=$BM/bias_model_$b/d0_all_fold_0/models/d0_all_fold_0_bias.h5
    [[ -f $f ]] && bias_models+=("bias_$b=$f")
done

echo "== predict"
legacy predict --out "$OUT" --genome "$GENOME" \
    --coords-h5 "$BM/bias_model_065/d0_all_fold_0/evaluation/d0_all_fold_0_bias_predictions.h5" \
    --models "${bias_models[@]}" "bias_model_scaled=$FM/models/bias_model_scaled.h5" \
             "chrombpnet=$FM/models/chrombpnet.h5" "chrombpnet_nobias=$FM/models/chrombpnet_nobias.h5"

echo "== shap bias_065 (counts + profile)"
legacy shap --out "$OUT" --genome "$GENOME" --name bias_065 --model "${bias_models[-1]#*=}" \
    --regions "$BM/bias_model_065/d0_all_fold_0/auxiliary/interpret_subsample/d0_all_fold_0_bias.interpreted_regions.bed" \
    --n 64 --heads counts profile
legacy fvals --out "$OUT" --name bias_065 --model "${bias_models[-1]#*=}"

echo "== shap chrombpnet_nobias (profile)"
legacy shap --out "$OUT" --genome "$GENOME" --name chrombpnet_nobias --model "$FM/models/chrombpnet_nobias.h5" \
    --regions "$FM/auxiliary/30K_subsample_peaks.bed" --n 16 --heads profile counts
legacy fvals --out "$OUT" --name chrombpnet_nobias --model "$FM/models/chrombpnet_nobias.h5"

echo "== traces"
legacy trace --out "$OUT" --name bias_128x4 --bigwig "$FM/auxiliary/data_unstranded.bw" \
    --filters 128 --n-dil-layers 4 --counts-loss-weight 10
legacy trace --out "$OUT" --name chrombpnet_32x4 --bigwig "$FM/auxiliary/data_unstranded.bw" \
    --filters 32 --n-dil-layers 4 --counts-loss-weight 10 --bias-model "$FM/models/bias_model_scaled.h5"

python3 - "$OUT" "$START" <<'PY'
import hashlib, json, os, sys
out, start = sys.argv[1], sys.argv[2]
files = {}
for root, _, names in os.walk(out):
    for n in names:
        p = os.path.join(root, n)
        if "/.mpl" in p or n == "manifest.json":
            continue
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        files[os.path.relpath(p, out)] = {"bytes": os.path.getsize(p), "md5": h}
json.dump({"started": start, "versions": json.load(open(os.path.join(out, "versions.json"))), "files": files},
          open(os.path.join(out, "manifest.json"), "w"), indent=1)
print("wrote manifest with", len(files), "files")
PY
du -sh "$OUT"
