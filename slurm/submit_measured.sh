#!/bin/bash
# Measured-energy round 1 on Leonardo Booster. Run on a LOGIN node.
#   bash slurm/submit_measured.sh          # check job, then 8 exclusive shards (4 datasets x cpu/gpu)
#   DRY=1 bash slurm/submit_measured.sh    # print the plan, submit nothing
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"   # cds into PROJ
mkdir -p logs/slurm

submit() {  # prints the job id; DRY=1 prints a fake one
  if [ "${DRY:-0}" = 1 ]; then echo "[dry] sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi
}

CHECK=$(submit slurm/measure_check.sbatch); echo "check   ${CHECK}"
for DEV in cpu gpu; do
  for D in synthetic ettm1 electricity solar; do
    J=$(submit --dependency=afterok:${CHECK} --job-name="jg_meas_${DEV}_${D}" slurm/measure.sbatch "${D}" "${DEV}")
    echo "${DEV} ${D}  ${J}"
  done
done
echo "logs: ${PROJ}/logs/slurm/   watch: squeue --me"
