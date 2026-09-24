# Changelog

## 2.0.0.dev0 (unreleased)

ChromBPNet moves from TensorFlow 2.8 / tf.keras to Keras 3 on the JAX backend, so it runs natively on CUDA 13 GPUs
(H100, B200, RTX PRO 6000 Blackwell). Commands, inputs and output files are unchanged except where noted below.

### Platform and packaging
- Keras 3 (>= 3.15.1) on JAX (>= 0.11.2). TensorFlow is no longer installed or imported. GPU support comes from
  `jax[cuda13]` (NVIDIA driver >= 580) or the `cuda12` fallback. The CUDA libraries are pip wheels: no system
  CUDA, no `module load cuda`, no `LD_LIBRARY_PATH`.
- Python >= 3.12 (3.13 in the lock files), NumPy 2, pandas 3, pyfaidx >= 0.8.1, weasyprint >= 66.
- `pyproject.toml` replaces `setup.py` and `requirements.txt`. pixi (`pixi.lock`) provides the environments
  `default`, `dev`, `cuda13`, `cuda13-dev` and `cuda12`, including bedtools, bedGraphToBigWig, samtools, MEME and
  pango. uv (`uv.lock`) provides the extras `cuda13` and `cuda12`.
- Removed dependencies: tensorflow, tensorflow-probability, protobuf, deepdish, deeplift, kundajelab-shap,
  modisco-lite 2.0.7, modisco 0.5.16 and matplotlib-venn.
- `modisco` >= 2.5.2 (tfmodisco-lite) replaces modisco-lite. The reports call `modisco report-simple`, because
  `modisco report` writes a different report in 2.5.
- The Docker image is rebuilt on debian:trixie-slim with pixi (`Dockerfile`; CUDA from the JAX wheels). It is
  published to `ghcr.io/nnfc-gmd/chrombpnet` instead of Docker Hub `kundajelab/chrombpnet`. New GitHub Actions CI
  checks the lock files, runs the CPU tests on Linux and macOS, runs lint, and tests a uv-only install.
- Job templates for SLURM GPU clusters (`workflows/slurm/`) and molab GPU sessions (`workflows/molab/`).

### Interpretation
- A native JAX DeepSHAP replaces kundajelab-shap (TF1 graph mode). The semantics are the same: the DeepLIFT
  rescale rule for ReLUs, 20 dinucleotide-shuffled references per sequence, the same profile-head weighting, and
  hypothetical scores projected onto the input. It is faster, batched on the GPU, and uses `highest` precision by
  default.
- DeepSHAP references are now seeded (`--shap-seed`, default 1234) from the seed and each sequence's content.
  Scores are reproducible and do not depend on the order or batching of regions. The 1.x references were
  unseeded, so old and new scores agree statistically, not exactly.
- Contribution-score files keep the 1.x layout: `/raw/seq` int8, `/shap/seq` and `/projected_shap/seq` float16,
  all (N, 4, L), Blosc-compressed. h5py + hdf5plugin now write them instead of deepdish.

### Behaviour changes
- EarlyStopping restores the weights of the best epoch even when training runs all `--epochs` without stopping
  early. 1.x kept the last epoch in that case.
- Initial weights and random streams differ from TensorFlow for the same `--seed`, so retrained models are
  statistically, not bitwise, equivalent to 1.x models. The seed now also controls the initialisation of the
  no-bias ChromBPNet model.
- New model files (`bias.h5`, `bias_model_scaled.h5`, `chrombpnet.h5`, `chrombpnet_nobias.h5`) keep their names
  but are written by Keras 3, so chrombpnet 1.x and TensorFlow 2.x cannot load them. chrombpnet 1.x models still
  load in 2.x: on load, a registered `LogSumExp` layer replaces the logsumexp `Lambda` of `chrombpnet.h5`.
- JAX allocates GPU memory on demand: `XLA_PYTHON_CLIENT_PREALLOCATE=false` unless you set it yourself.
- Removed: the legacy modisco-0.5 scripts (`evaluation/modisco/{run_modisco,fetch_tomtom,visualize_motif_matches}.py`,
  `modisco.sh`) and `evaluation/invivo_footprints/`.

### New optional flags (the defaults keep the 1.x behaviour)
- Training: `--optimizer {adam,muon}`, `--muon-lr`, `--ema`, `--lr-schedule {constant,cosine}`,
  `--precision {default,highest,bf16}`, `--device {auto,gpu,cpu}`.
- Interpretation: `--interpret-subsample` (default 30000), `--shap-seed` (default 1234), `--shap-batch-seqs`,
  `--shap-precision {highest,default}`.
- MoDISco: `--modisco-max-seqlets` (default 50000), `--modisco-window` (default 500), `--tomtom-lite`.

##  Version - 1.5
- Fixed issue #150, regions_used not found while generating bigwigs from impotance h5s

## Version - 1.4
- (MAJOR) Bug in chrombpnet modisco_motifs command. seqlets was limited to 50000. If users wanted to change it to 1 million this did not happen.
- Filter peaks at edges for pred_bw command and bias pipleline. So bias evaluation now done on these filtered peaks.
- Preprocessing deafulted to use unix sort. Provided option to switch to bedtools sort.
- Provided option to use filter chromosomes option in preprocessing.

## Version - 1.3 - Inworks - 2022-12-11
- Changed pipelines to use modisco-lite, old modisco will soon be removed
- Added automatic shift scripts to repo and integrated with the pipeline
- modisco now outputs both html and pdf. pdf can be shared with anyone.
- Use ATAC, DNASE not ATAC_PE, DNASE_SE anymore.
- Simplyfing workflows to include only two main workflows, chrombpnet_train_tf_model, chrombpnet_train_bias_model
- Restructuring README, moving tutorial to additional documentation and introducing FAQ, and only two pipelines (chrombpnet_train_tf_model, chrombpnet_train_bias_model)

## [Unreleased] - 2022-02-28
- Typo fix - (PR#31-36)
- We can now specify ylimit in marginal footprinting plots (PR#27)
- PR#30 merged  to do fast genome wide gc binning and bug fix to ensure case-sensitve GC calculation. Adds a unit test to make checks.

## [Unreleased] - 2022-02-11
- Updated modisco version - (PR#25)

## [Unreleased] - 2022-02-07
- Typo fix - (PR#23-24)

## [Unreleased] - 2022-01-31
- Marginal footprinting no longer hardcoded to take only chr1, it now inputs the fold json. (PR#20)
- Invivo footprinting is made compatible with the new input repo changes. (PR#22)
- Pandas read bugs fixed for gc-matching scripts. (PR#15)
- Tutorial now exits when one part of the pipeline breaks instead of proceeding further. (PR#18)
- Added the fast binning scripts at src/helpers/make_gc_matched_negatives/get_genomewide_gc_buckets (PR#11)
- Added pseudocount in metrics.py (PR#13) - needs further review.

## [Unreleased] - 2022-01-24
- Seed setting fixed for training. Seed was set improperly - preventing shuffling for every epoch. (PR#19)
- return coordinates while training for debugging - returns status of the point such as revcomp,peak/nonpeak,coordinates etc

## [Unreleased] - 2022-01-15
- In `step6_train_chrombpnet_model` changed the file saving name bug  
- Changed scripts in `src/evaluatuion/make_bigwigs/` to be compatible with new dir structure

## [Unreleased] - 2022-01-13
- Added a note in README that users can continue to use the pre-trained bias model from chrombpnet-lite repo, if thats what they have been using till now
- Changed the default modisco crop setting in tutorial from 1000 to 500
- Fixed a bug in `src/evaluation/modisco/run_modisco.py` in directory creation
