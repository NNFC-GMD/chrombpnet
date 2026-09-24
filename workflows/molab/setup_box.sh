#!/usr/bin/env bash
# Install (or re-check) the chrombpnet pixi environment in the existing /marimo/chrombpnet checkout, then check
# the GPU and print versions. Idempotent: re-run it after every new molab session.
#
#   molab sbatch -D /marimo/chrombpnet --rc /marimo/chrombpnet/workflows/molab/env.sh -c 2 -t 45 -J cbp_setup \
#       --wrap 'bash workflows/molab/setup_box.sh'
#   molab tail -f <jobid>
#
# Exit status: 0 ready, 1 install failed, 2 installed but JAX sees no GPU (see `molab sinfo`).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Also works without --rc.
# shellcheck source-path=SCRIPTDIR source=env.sh
source "${here}/env.sh"
repo="${CHROMBPNET_REPO}"
env_name="${CHROMBPNET_PIXI_ENV}"
pixi_version="${PIXI_VERSION:-0.81.0}"
cd "${repo}"

echo "== chrombpnet checkout ${repo}: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') @ $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "== scratch ${CHROMBPNET_SCRATCH}, pixi env ${env_name}, threads ${OMP_NUM_THREADS}, GPU memory fraction ${XLA_PYTHON_CLIENT_MEM_FRACTION}"

if ! command -v pixi >/dev/null 2>&1; then
    echo "== pixi not found: installing ${pixi_version} into ${CHROMBPNET_SCRATCH}/bin"
    mkdir -p "${CHROMBPNET_SCRATCH}/bin"
    curl -fsSL -o "${CHROMBPNET_SCRATCH}/bin/pixi.tmp" \
        "https://github.com/prefix-dev/pixi/releases/download/v${pixi_version}/pixi-$(uname -m)-unknown-linux-musl"
    chmod 0755 "${CHROMBPNET_SCRATCH}/bin/pixi.tmp"
    mv "${CHROMBPNET_SCRATCH}/bin/pixi.tmp" "${CHROMBPNET_SCRATCH}/bin/pixi"
    export PATH="${CHROMBPNET_SCRATCH}/bin:${PATH}"
fi
echo "== $(pixi --version)"

if gpu=$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader 2>/dev/null) \
        && [[ -n "${gpu}" ]]; then
    echo "== GPU: ${gpu}"
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)
    if [[ "${env_name}" == cuda13* && "${driver_major}" -lt 580 ]]; then
        echo "!! driver ${driver_major}.x is older than 580, which CUDA 13 needs: use CHROMBPNET_PIXI_ENV=cuda12"
    fi
else
    echo "!! WARNING: no GPU visible (nvidia-smi failed or lists none). A recreated molab session can come back"
    echo "!!          without its GPU: check 'molab sinfo'. Installing anyway; gpu-check below will fail."
fi

# The cuda envs require the __cuda virtual package; the override lets the install proceed even when the driver
# is not visible (the gpu-check below is what tells whether the GPU works). --locked never rewrites pixi.lock.
echo "== pixi install --locked -e ${env_name}"
if ! CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-13.0}" pixi install --locked -e "${env_name}"; then
    echo "!! pixi install failed"
    exit 1
fi

status=0
echo "== versions"
pixi run -e "${env_name}" --frozen versions || status=1
echo "== tools"
# shellcheck disable=SC2016  # expanded by the inner bash
pixi run -e "${env_name}" --frozen bash -c \
    'set -e; bedtools --version; samtools --version | head -n1; printf "tomtom "; tomtom -version; modisco --help >/dev/null; echo "modisco ok"' \
    || status=1
echo "== gpu-check"
if ! pixi run -e "${env_name}" --frozen gpu-check; then
    echo "!! gpu-check FAILED: JAX sees no CUDA device. Check 'molab sinfo' and the NVIDIA driver (>= 580 for cuda13)."
    status=2
fi
du -sh ".pixi/envs/${env_name}" "${CHROMBPNET_SCRATCH}/cache" 2>/dev/null || true
if [[ ${status} -eq 0 ]]; then
    echo "== ready"
fi
exit ${status}
