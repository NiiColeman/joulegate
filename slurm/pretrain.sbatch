#!/bin/bash
# Pretrain the frozen backbone -> checkpoints/{backbone.pt, retention_windows.npy}.
# About 15 min on 2 cores per the README; 8 cores here.
#SBATCH --job-name=jg_pretrain
#SBATCH --account=iscrb_collage
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

${PY} -m scripts.pretrain "$@"
