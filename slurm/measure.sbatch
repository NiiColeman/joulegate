#!/bin/bash
# One measured-energy shard: every configuration of a sweep file for the given dataset(s) on
# one device, one run at a time on an exclusive node (RAPL sees the whole socket).
#   sbatch slurm/measure.sbatch <dataset[,dataset...]> <cpu|gpu> [sweep file] [results root]
# Defaults: configs/sweeps/measured.yaml and results/measured/<device>.
#SBATCH --job-name=jg_measure
#SBATCH --account=iscrb_collage
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

USAGE="usage: sbatch slurm/measure.sbatch <dataset[,dataset...]> <cpu|gpu> [sweep file] [results root]"
DATASETS=${1:?${USAGE}}
DEV=${2:?${USAGE}}
SPEC=${3:-configs/sweeps/measured.yaml}
OUT=${4:-results/measured}
case ${DEV} in
  cpu) TORCH=cpu ;;
  gpu) TORCH=cuda ;;
  *) echo "device must be cpu or gpu"; exit 1 ;;
esac
echo "=== measured ${DATASETS} on ${DEV} (${SPEC}), $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi -L
# EXTRA_SET: optional space-separated overrides for every run, e.g. EXTRA_SET="threads=8"
${PY} sweep.py "${SPEC}" --datasets ${DATASETS//,/ } --workers 1 \
  --energy "measured_${DEV}" --out "${OUT}/${DEV}" --set "device=${TORCH}" loop.buffer_eval=true ${EXTRA_SET:-}
