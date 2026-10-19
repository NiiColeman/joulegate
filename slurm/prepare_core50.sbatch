#!/bin/bash
# Extract and convert the downloaded CORe50 zip (data/core50_128x128.zip, fetched on the login
# node with CORE50_DOWNLOAD_ONLY=1) into per-session arrays in data/core50/.
#SBATCH --job-name=jg_prepare_core50
#SBATCH --account=iscrb_collage
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
test -f data/core50_128x128.zip || { echo "download the zip on the login node first"; exit 1; }
${PY} -m datasets.download core50
