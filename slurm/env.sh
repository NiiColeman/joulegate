# Shared environment for every JouleGate Slurm job. Sourced, not executed.
# PROJ is the repo this file sits in. JOULEGATE_ENV overrides the Python env (default: the Leonardo one);
# the Slurm account is set by #SBATCH --account in each job and can be overridden with sbatch --account=<yours>.
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV=${JOULEGATE_ENV:-/leonardo_work/IscrB_COLLAGE/1c3c0l3d/envs/glam311}
PY=${ENV}/bin/python

# The code is CPU-only (analytic energy meter); each run sets torch threads itself
# (threads: 1 in configs/base.yaml), so stop BLAS/OpenMP from spawning more.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # compute nodes have no internet; weights are cached from the login node
cd "${PROJ}"
