#!/bin/bash
# PACS / Office-Home end to end: convert the downloaded parquets (fetch them on the login node with
# DOWNLOAD_ONLY=1 python -m datasets.download <name>), fine-tune on the base domain, run the sweep.
#   sbatch slurm/multi_domain.sbatch pacs
#SBATCH --job-name=jg_domains
#SBATCH --account=iscrb_collage
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
NAME=${1:?"usage: sbatch slurm/multi_domain.sbatch <pacs|officehome>"}
test -d "data/${NAME}" || ${PY} -m datasets.download "${NAME}"
${PY} -m scripts.pretrain_domains "${NAME}" --device cuda
${PY} sweep.py "configs/sweeps/${NAME}.yaml" --workers "${SLURM_CPUS_PER_TASK}"
