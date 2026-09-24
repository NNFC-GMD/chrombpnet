# shellcheck shell=bash
# Job environment for chrombpnet on a molab box. molab sources it before every job:
#   molab sbatch -D /marimo/chrombpnet --rc /marimo/chrombpnet/workflows/molab/env.sh -c 2 -t 60 --wrap '...'
# Every value is only a default, so `--export VAR=value` (or an exported variable) overrides it. It never reads
# secrets and writes nothing outside ${CHROMBPNET_SCRATCH}. Do not `set -e/-u` here: this file is sourced into
# the job's own shell.

# The notebook kernel's Python must not leak into jobs (Apptainer binds /tmp, where its venv lives).
unset PYTHONPATH PYTHONHOME PYTHONSAFEPATH VIRTUAL_ENV
# JAX uses the CUDA 13 libraries of its pip wheels; anything on LD_LIBRARY_PATH would shadow them.
unset LD_LIBRARY_PATH

export CHROMBPNET_REPO="${CHROMBPNET_REPO:-/marimo/chrombpnet}"
export CHROMBPNET_SCRATCH="${CHROMBPNET_SCRATCH:-/marimo/chrombpnet/.scratch}"
export CHROMBPNET_PIXI_ENV="${CHROMBPNET_PIXI_ENV:-cuda13-dev}"
# setup_box.sh puts a pinned pixi here when the box has none.
if [[ -x "${CHROMBPNET_SCRATCH}/bin/pixi" ]]; then
    export PATH="${CHROMBPNET_SCRATCH}/bin:${PATH}"
fi

export KERAS_BACKEND=jax
# The GPU is shared with other jobs: allocate on demand, and cap this process at a fraction of the card's TOTAL
# memory (0.25 of the RTX PRO 6000's 96 GB is ~24 GB). Use 0.05 for smoke tests, more when the box is idle.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# The box reports the host's cores (20+), not the ~4 we get: size thread pools to what the job asked for (-c).
_cbp_threads="${CHROMBPNET_THREADS:-${SLURM_CPUS_PER_TASK:-2}}"
export OMP_NUM_THREADS="${_cbp_threads}"
export OPENBLAS_NUM_THREADS="${_cbp_threads}"
export MKL_NUM_THREADS="${_cbp_threads}"
export NUMEXPR_NUM_THREADS="${_cbp_threads}"
export NUMBA_NUM_THREADS="${_cbp_threads}"
unset _cbp_threads

# Caches live under /marimo (a new session keeps /marimo, not $HOME) and on the same filesystem as .pixi/, so
# pixi can hardlink packages instead of copying them.
_cbp_cache="${CHROMBPNET_SCRATCH}/cache"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-${_cbp_cache}/rattler}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${_cbp_cache}/uv}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${_cbp_cache}/jax}"
export KERAS_HOME="${KERAS_HOME:-${_cbp_cache}/keras}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${_cbp_cache}/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${_cbp_cache}/numba}"
mkdir -p "${PIXI_CACHE_DIR}" "${UV_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}" "${KERAS_HOME}" "${MPLCONFIGDIR}" \
    "${NUMBA_CACHE_DIR}"
unset _cbp_cache

export PYTHONUNBUFFERED=1
