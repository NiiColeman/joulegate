#!/bin/bash
# Full JouleGate pipeline on Leonardo Booster. Run on a LOGIN node (compute nodes have no internet).
#
#   bash slurm/submit.sh                       # tests -> main/conservative/depth sweeps -> analyze
#   bash slurm/submit.sh --pretrain            # also retrain the backbone first
#   bash slurm/submit.sh main dev              # only these sweeps (configs/sweeps/<name>.yaml)
#   DRY=1 bash slurm/submit.sh                 # print the plan, submit nothing
#
# Stages, chained with afterok:  [pretrain] -> tests -> sweep array (one task per spec x dataset) -> analyze
# analyze runs only when the default three sweeps are all requested (it needs all three result trees).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"   # cds into PROJ

PRETRAIN=0
SPECS=()
for a in "$@"; do
  case $a in
    --pretrain) PRETRAIN=1 ;;
    -*) echo "unknown flag $a"; exit 1 ;;
    *) SPECS+=("$a") ;;
  esac
done
[ ${#SPECS[@]} -eq 0 ] && SPECS=(main conservative depth)
mkdir -p logs/slurm

# 1. Data: download (login node only) and warm data/cache so parallel workers never race on np.save.
${PY} -m datasets.download
${PY} - <<'EOF'
import glob, os, yaml
from datasets.streams import load_matrix
files = {yaml.safe_load(open(f)).get("file") for f in glob.glob("configs/datasets/*.yaml")}
files |= {s["file"] for s in yaml.safe_load(open("configs/pretrain.yaml"))["sources"]}
for name in sorted(f for f in files if f and f != "synthetic"):
    print(f"cache {name}: {load_matrix(name).shape}")
EOF

# 2. Shards: one line "<spec> <dataset>" per array task, skipping shards with nothing left to run.
SHARDS=logs/slurm/shards_$(date +%Y%m%d_%H%M%S).txt
${PY} - "${SPECS[@]}" > "${SHARDS}" <<'EOF'
import sys, yaml
from sweep import expand
for name in sys.argv[1:]:
    path = f"configs/sweeps/{name}.yaml"
    spec = yaml.safe_load(open(path))
    for d in spec["datasets"]:
        n = len(expand(spec, [d]))
        print(f"  {path} {d}: {n} runs pending", file=sys.stderr)
        if n:
            print(path, d)
EOF
N=$(wc -l < "${SHARDS}")
echo "${N} shards -> ${SHARDS}"

submit() {  # prints the job id; DRY=1 prints a fake one
  if [ "${DRY:-0}" = 1 ]; then echo "[dry] sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi
}
dep() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }

PRE=""
if [ "${PRETRAIN}" = 1 ] || [ ! -f checkpoints/backbone.pt ]; then
  PRE=$(submit slurm/pretrain.sbatch); echo "pretrain  ${PRE}"
fi
TEST=$(submit $(dep "${PRE}") slurm/tests.sbatch); echo "tests     ${TEST}"

LAST=${TEST}
if [ "${N}" -gt 0 ]; then
  SWEEP=$(submit --dependency=afterok:${TEST} --array=0-$((N - 1)) slurm/sweep.sbatch "${SHARDS}")
  echo "sweep     ${SWEEP} (array 0-$((N - 1)))"
  LAST=${SWEEP}
fi
if [ "$(printf '%s\n' "${SPECS[@]}" | sort | tr '\n' ' ')" = "conservative depth main " ]; then
  AN=$(submit --dependency=afterok:${LAST} slurm/analyze.sbatch); echo "analyze   ${AN}"
fi
echo "logs: ${PROJ}/logs/slurm/   watch: squeue --me"
