# ChromBPNet on a molab GPU box

Scripts for running this repository on a [molab](https://molab.marimo.io) session through
[molab-slurm](https://broadinstitute.github.io/molab-slurm/) (`molab sbatch`, `molab srun`, ...). The box used for
development is Debian 13 with one RTX PRO 6000 Blackwell (96 GB, sm_120) and NVIDIA driver 595 (CUDA 13.2). The
real slice is about 4 CPUs and 32 GB of RAM, even though the box reports more.

| file | what it does |
|---|---|
| `env.sh` | job environment, passed to every job with `--rc`: JAX backend and GPU-sharing limits, thread counts, caches under `/marimo` |
| `setup_box.sh` | idempotent setup: `pixi install --locked -e cuda13-dev` in `/marimo/chrombpnet`, GPU check, versions. Safe to re-run |

Everything lives in the existing checkout `/marimo/chrombpnet`. Run directories, caches and goldens go to
`/marimo/chrombpnet/.scratch/` (git-ignored). Never source another project's `env.sh`. The igvf one, for
example, writes git config and exports secrets.

## After every new session

A session can end at any time: the 12-hour limit, idle shutdown, a crash, or running out of memory or disk.
molab may then give you a **new** box that keeps only part of `/marimo`. Jobs that were running are gone, and so
is molab-slurm's own state.

1. Reconnect: `molab init <new-url> <token> --name gpu` (same name, so scripts keep working).
2. `molab sinfo`. **Check that the GPU is there.** A recreated session can come back without it, and JAX then
   quietly runs on the CPU.
3. Check that the checkout is still at the expected commit, and ship it again if not (see below).
4. Re-run the setup. It is quick when `.pixi/` and `.scratch/cache/` survived:

   ```bash
   molab sbatch -D /marimo/chrombpnet --rc /marimo/chrombpnet/workflows/molab/env.sh -c 2 -t 45 -J cbp_setup \
       --wrap 'bash workflows/molab/setup_box.sh'
   molab tail -f <jobid>
   ```

   Exit status 0 means ready, 1 means the install failed, and 2 means the env is installed but JAX sees no GPU.
   Long commands go through `sbatch` + `tail`, not `srun`: kernel replies are capped at about 1 MB, and requests
   that run for minutes can come back empty.
5. Resubmit your jobs. `chrombpnet pipeline` / `bias pipeline` do not resume and refuse an existing output
   directory, so move the partial run aside or use a fresh `.scratch/runs/<name>`.

### Shipping unpushed commits

The box already has the fork's history, so an incremental bundle stays far below `molab put`'s 64 MB limit:

```bash
git bundle create /tmp/cbp.bundle <commit-already-on-the-box>..h100-support
molab put /tmp/cbp.bundle /tmp/
molab srun -D /marimo/chrombpnet -t 5 git fetch /tmp/cbp.bundle h100-support
molab srun -D /marimo/chrombpnet -t 5 git checkout -B h100-support FETCH_HEAD
```

Do not print the checkout's remote URL on the box (it can embed a credential).

## Running jobs

Always pass `-D /marimo/chrombpnet`, the absolute `--rc`, a walltime `-t`, and a CPU count `-c` (it sizes
the thread pools, see `env.sh`). Use `pixi run --frozen`, which never rewrites `pixi.lock` on the box.

```bash
RC=/marimo/chrombpnet/workflows/molab/env.sh

# GPU test suite, CPU test suite
molab sbatch -D /marimo/chrombpnet --rc $RC -c 2 -t 30 -J cbp_test_gpu --wrap 'pixi run -e cuda13-dev --frozen test-gpu'
molab sbatch -D /marimo/chrombpnet --rc $RC -c 2 -t 30 -J cbp_test --wrap 'pixi run -e cuda13-dev --frozen test'

# A pipeline run (fails fast with --device gpu if the session came back without a GPU)
molab sbatch -D /marimo/chrombpnet --rc $RC -c 4 -t 12:00:00 -J cbp_pipeline --wrap '
  pixi run -e cuda13 --frozen chrombpnet pipeline --device gpu \
    -ifrag /marimo/data/<dataset>/fragments.tsv.gz -d ATAC \
    -g /marimo/data/references/hg38/Sequence/<genome>.fasta -c <chrom.sizes> \
    -p <peaks.narrowPeak> -n <nonpeaks.bed> -fl <fold_0.json> -b <bias.h5> \
    -o /marimo/chrombpnet/.scratch/runs/<name>'
```

Override any `env.sh` default per job with `--export`, for example
`--export XLA_PYTHON_CLIENT_MEM_FRACTION=0.05` for a smoke test or `--export CHROMBPNET_PIXI_ENV=cuda12`.

## Sharing the GPU

Several jobs can use the card at the same time. molab does not enforce `--mem`/`--gres`, and the kernel's OOM
killer picks the largest process, which is usually someone's long training job. To avoid that:

* `XLA_PYTHON_CLIENT_PREALLOCATE=false` (in `env.sh`, and chrombpnet's own default): JAX allocates on demand
  instead of taking 75% of the card at start-up.
* `XLA_PYTHON_CLIENT_MEM_FRACTION` (default 0.25 here, about 24 GB) caps the pool even without preallocation.
  It is a fraction of the card's **total** memory, not of what is free. Check `nvidia-smi` before raising it.
* While other people's jobs are running, use `-c 1`-`2`, keep host memory under about 6 GB, and put heavy work
  behind their last job: `--dependency=afterany:<jobid>`. Run timing benchmarks only when `molab squeue` is
  empty and the GPU is idle.
* The JAX CPU backend (XLA) ignores `OMP_NUM_THREADS`, so run CPU-only tests with a small `-c` anyway.

## Environment reference (`env.sh`)

| variable | default | notes |
|---|---|---|
| `CHROMBPNET_REPO` | `/marimo/chrombpnet` | checkout the setup installs into |
| `CHROMBPNET_SCRATCH` | `/marimo/chrombpnet/.scratch` | runs, caches, goldens |
| `CHROMBPNET_PIXI_ENV` | `cuda13-dev` | env `setup_box.sh` installs (`cuda13`, `cuda13-dev`, `cuda12`) |
| `CHROMBPNET_THREADS` | `$SLURM_CPUS_PER_TASK`, else 2 | OMP/OpenBLAS/MKL/numexpr/numba threads |
| `KERAS_BACKEND` | `jax` | always |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.25` | |
| `CUDA_VISIBLE_DEVICES` | `0` | |
| `PIXI_CACHE_DIR`, `UV_CACHE_DIR` | `.scratch/cache/{rattler,uv}` | same filesystem as `.pixi/`, so installs hardlink |
| `JAX_COMPILATION_CACHE_DIR` | `.scratch/cache/jax` | persistent XLA compilation cache |
| `KERAS_HOME`, `MPLCONFIGDIR`, `NUMBA_CACHE_DIR` | `.scratch/cache/{keras,matplotlib,numba}` | |

`env.sh` also unsets `PYTHONPATH`, `PYTHONHOME`, `PYTHONSAFEPATH`, `VIRTUAL_ENV` (the notebook kernel's venv)
and `LD_LIBRARY_PATH`, because JAX must use the CUDA libraries from its own wheels. The disk quota on the box is
hidden. `du -sh /marimo/chrombpnet/.pixi /marimo/chrombpnet/.scratch/cache` shows what the setup uses, and
`pixi clean cache` reclaims the package cache.
