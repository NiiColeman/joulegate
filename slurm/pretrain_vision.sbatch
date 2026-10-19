#!/bin/bash
# Fine-tune ViT-Tiny on CORe50 sessions 1-2 -> checkpoints/vit_tiny_core50.pt (+ retention set).
# Other source sessions: sbatch slurm/pretrain_vision.sbatch --base 1 2 3 --suffix _b3
#SBATCH --job-name=jg_pretrain_vision
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
${PY} -m scripts.pretrain_vision --device cuda "$@"
