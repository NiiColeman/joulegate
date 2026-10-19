#!/bin/bash
# Wall-clock of one 800-step CORe50 dev run per setting, to size the vision sweeps and measured jobs.
#SBATCH --job-name=jg_vision_timing
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
rm -rf results/timing
t() {  # label, energy config, then extra --set values (run.py keeps only one --set list)
  local label=$1 energy=$2; shift 2; local t0=$(date +%s)
  ${PY} run.py --dataset core50_dev --method periodic --energy "${energy}" --out results/timing \
    --set seed=0 method.period=4 tag="${label}" "$@"
  echo "TIMING ${label}: $(( $(date +%s) - t0 )) s"
}
# analytic_cpu_t1 took 1487 s (job 59416214); a 1-thread vision run outlasts fake.yaml's 600 s
# sampler capacity, so the measured-style runs get more.
t fake_cpu_t8 fake threads=8 energy.max_seconds=7200
t fake_gpu fake device=cuda energy.max_seconds=7200
t fake_cpu_t1 fake threads=1 energy.max_seconds=7200
