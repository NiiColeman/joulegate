#!/bin/bash
# Measured-energy round 2 on Leonardo Booster. Run on a LOGIN node.
#   CPU: the full main grid, 3 seeds (configs/sweeps/measured_full.yaml), one exclusive job per stream
#   GPU: exact per-call joules (forecast, probe, update) from one frozen run per stream, one job
#   DRY=1 bash slurm/submit_measured_round2.sh    # print the plan, submit nothing
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"   # cds into PROJ
mkdir -p logs/slurm

submit() {  # prints the job id; DRY=1 prints a fake one
  if [ "${DRY:-0}" = 1 ]; then echo "[dry] sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi
}

CHECK=$(submit slurm/measure_check.sbatch); echo "check   ${CHECK}"
for D in synthetic ettm1 electricity solar; do
  J=$(submit --dependency=afterok:${CHECK} --time=03:30:00 --job-name="jg_full_cpu_${D}" \
      slurm/measure.sbatch "${D}" cpu configs/sweeps/measured_full.yaml results/measured_full)
  echo "cpu full ${D}  ${J}"
done
J=$(submit --dependency=afterok:${CHECK} --time=00:30:00 --job-name=jg_percall_gpu \
    slurm/measure.sbatch synthetic,ettm1,electricity,solar gpu configs/sweeps/measured_percall.yaml results/measured_percall)
echo "gpu per-call  ${J}"
echo "logs: ${PROJ}/logs/slurm/   watch: squeue --me"
