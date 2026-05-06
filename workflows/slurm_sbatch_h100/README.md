# H100 Slurm Helpers

These scripts assume the trained model directory created by
`chrombpnet_pipeline_h100.sbatch`:

```bash
BASE=/dcai/users/mateug/chrombpnet_tutorial
OUT=$BASE/chrombpnet_model
```

Submit the prediction bigWig job with the H100 TensorFlow environment:

```bash
sbatch workflows/slurm_sbatch_h100/chrombpnet_pred_bw_h100.sbatch
```

`chrombpnet_pred_bw_h100.sbatch` defaults `OBSERVED_BW` to
`$OUT/auxiliary/data_unstranded.bw`. If the observed insertion/head-count
track lives somewhere else, override it at submit time:

```bash
sbatch --export=OBSERVED_BW=/path/to/head.bw,TRACK_PREFIX=head \
  workflows/slurm_sbatch_h100/chrombpnet_pred_bw_h100.sbatch
```

## H100 Environment Snapshot

`chrombpnet-h100.full.yml` and `chrombpnet-h100.pip-freeze.txt` are snapshots
of the working H100 training/prediction environment. They intentionally include
TensorFlow 2.21 and NumPy 2.4, so this is not the DeepSHAP environment.

To recreate the same H100 environment at the exported cluster path:

```bash
conda env create -f workflows/slurm_sbatch_h100/chrombpnet-h100.full.yml

conda activate /dcai/users/mateug/envs/chrombpnet-h100
cd /dcai/users/mateug/git/chrombpnet
git checkout h100-support
git pull --ff-only
python -m pip install --no-deps -e .
```

The `pip-freeze` file is for auditing exact resolved packages. Prefer the
editable install command above over installing directly from the freeze file,
because the freeze captures a Git revision for `chrombpnet` while active
development is happening on the `h100-support` branch.

DeepSHAP is intentionally split into a separate job because the legacy
`kundajelab-shap` stack is fragile with NumPy 2 and newer IPython. Use a
separate interpretation environment. This environment keeps TensorFlow 2.21 but
pins NumPy and IPython around the older SHAP code:

```bash
conda env create -p /dcai/users/mateug/envs/chrombpnet-shap \
  -f workflows/slurm_sbatch_h100/chrombpnet_deepshap_tf221_env.yml

conda activate /dcai/users/mateug/envs/chrombpnet-shap
python -m pip install --no-deps -e /dcai/users/mateug/git/chrombpnet

python - <<'PY'
import numpy as np
import IPython
import tensorflow as tf
import pyBigWig
import shap
print("NumPy", np.__version__)
print("IPython", IPython.__version__)
print("TensorFlow", tf.__version__)
print("pyBigWig OK")
print("SHAP OK")
PY
```

If pip upgraded NumPy during environment creation, repair the existing env with:

```bash
conda activate /dcai/users/mateug/envs/chrombpnet-shap
python -m pip install --force-reinstall --no-cache-dir "numpy==1.26.4" "ipython<9"
python -m pip install --no-deps -e /dcai/users/mateug/git/chrombpnet
```

Then submit:

```bash
sbatch --export=NIL workflows/slurm_sbatch_h100/chrombpnet_deepshap_legacy.sbatch
```

The DeepSHAP and MoDISco scripts request 64 GB by default. If a full run runs
out of memory, resubmit with a larger Slurm memory request, for example:

```bash
sbatch --mem=96G --export=NIL workflows/slurm_sbatch_h100/chrombpnet_deepshap_legacy.sbatch
```

If Slurm holds the job with `user env retrieval failed requeued held`, cancel
the held job and submit again without exporting the full login environment:

```bash
scancel JOBID
unset SBATCH_GET_USER_ENV SBATCH_EXPORT SLURM_EXPORT_ENV
sbatch --export=NIL workflows/slurm_sbatch_h100/chrombpnet_deepshap_legacy.sbatch
```

For a small DeepSHAP smoke test on clusters where variable export triggers
user-env retrieval, make a temporary script with fewer regions:

```bash
cp workflows/slurm_sbatch_h100/chrombpnet_deepshap_legacy.sbatch /tmp/deepshap_smoke.sbatch
sed -i 's/N_REGIONS=${N_REGIONS:-30000}/N_REGIONS=1000/' /tmp/deepshap_smoke.sbatch
sbatch --export=NIL /tmp/deepshap_smoke.sbatch
```

The sbatch scripts derive their temporary directory from `SLURM_JOB_USER` and
`SLURM_JOB_ID`, so they also work when `--export=NIL` strips login variables
such as `USER`.

Run MoDISco after DeepSHAP creates `profile_scores.h5` and `counts_scores.h5`:

```bash
sbatch workflows/slurm_sbatch_h100/chrombpnet_modisco.sbatch
```

Useful overrides:

```bash
sbatch --mem=96G --export=NIL workflows/slurm_sbatch_h100/chrombpnet_deepshap_legacy.sbatch
sbatch --export=NIL workflows/slurm_sbatch_h100/chrombpnet_modisco.sbatch
```
