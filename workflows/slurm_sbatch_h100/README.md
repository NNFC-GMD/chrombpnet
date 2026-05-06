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
sbatch --export=ALL,OBSERVED_BW=/path/to/head.bw,TRACK_PREFIX=head \
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
```

Then submit:

```bash
sbatch --export=ALL,SHAP_ENV=/dcai/users/mateug/envs/chrombpnet-shap \
  workflows/slurm_sbatch_h100/chrombpnet_deepshap_legacy.sbatch
```

The DeepSHAP and MoDISco scripts request 64 GB by default. If a full run runs
out of memory, resubmit with a larger Slurm memory request, for example:

```bash
sbatch --mem=96G --export=ALL,SHAP_ENV=/dcai/users/mateug/envs/chrombpnet-shap \
  workflows/slurm_sbatch_h100/chrombpnet_deepshap_legacy.sbatch
```

Run MoDISco after DeepSHAP creates `profile_scores.h5` and `counts_scores.h5`:

```bash
sbatch workflows/slurm_sbatch_h100/chrombpnet_modisco.sbatch
```

Useful overrides:

```bash
sbatch --export=ALL,N_REGIONS=1000 workflows/slurm_sbatch_h100/chrombpnet_deepshap_legacy.sbatch
sbatch --export=ALL,RUN_COUNTS=0 workflows/slurm_sbatch_h100/chrombpnet_modisco.sbatch
```
