#!/bin/bash
# Gate for the pipeline: energy unit tests plus one short end-to-end run.
#SBATCH --job-name=jg_tests
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

${PY} -m tests.test_energy
# Smoke run into a throwaway tree so it never satisfies a real sweep's skip check.
${PY} run.py --dataset ettm1 --method vou --out results/smoke --set seed=0 tag=smoke
