#!/bin/bash
# Figures and tables into paper/ from results/main, results/conservative and results/depth.
#SBATCH --job-name=jg_analyze
#SBATCH --account=iscrb_collage
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

${PY} -m analysis.analyze
${PY} -m analysis.compare_regimes
