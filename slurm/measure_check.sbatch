#!/bin/bash
# Gate for the measured round: the test suite on a GPU node, then a 50-step measured run on
# CPU and on GPU (not exclusive, so RAPL values include other jobs: plumbing check only).
#SBATCH --job-name=jg_measure_check
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

${PY} -m pytest -q tests/
${PY} -m tests.test_energy
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi -L
# A whole one-dataset sweep per device: consecutive runs in fresh workers, as in the real shards.
# Fresh output every time: sweep.py skips runs that already have a summary.
rm -rf results/measured_check
for DEV in cpu gpu; do
  TORCH=$([ "${DEV}" = gpu ] && echo cuda || echo cpu)
  ${PY} sweep.py configs/sweeps/measured.yaml --datasets ettm1 --workers 1 --energy "measured_${DEV}" \
    --out "results/measured_check/${DEV}" --set dataset.n_steps=50 "device=${TORCH}" loop.buffer_eval=true \
    energy.require_exclusive=false energy.pin=null energy.idle_s=5
done
${PY} - <<'PYEOF'
import glob, json
for f in sorted(glob.glob("results/measured_check/*/ettm1/*/seed0/summary.json")):
    s = json.load(open(f)); e = s["energy"]
    print(f.split("/")[1], f.split("/")[3], "updates", s["n_updates"], "closure", f"{e['closure']:.1e}",
          "idle W", {k: round(v, 1) for k, v in e["p_idle_w"].items()}, "drift", round(e["idle_drift"], 3),
          "adapt_j", round(s["adapt_j"], 3), "raw", round(s["adapt_j_raw"], 3), "price lag", s["price_lag_mean"])
PYEOF
