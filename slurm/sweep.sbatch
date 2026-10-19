#!/bin/bash
# One sweep shard = (sweep spec, dataset). Array task i runs line i of the shard file:
#   configs/sweeps/main.yaml synthetic
# sweep.py skips runs whose summary.json exists, so a timed-out shard can be resubmitted as is.
#SBATCH --job-name=jg_sweep
#SBATCH --account=iscrb_collage
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

SHARDS=${1:?"usage: sbatch --array=0-N slurm/sweep.sbatch <shard file>"}
read -r SPEC DATASET < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SHARDS}")
echo "=== shard ${SLURM_ARRAY_TASK_ID}: ${SPEC} ${DATASET} on $(hostname), ${SLURM_CPUS_PER_TASK} workers ==="
${PY} sweep.py "${SPEC}" --datasets "${DATASET}" --workers "${SLURM_CPUS_PER_TASK}"
