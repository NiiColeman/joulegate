# JouleGate

**When is it worth spending energy to update a deployed model?**

A pretrained model runs on a data stream that changes over time. It can be fine-tuned online (LoRA adapters, small buffer of recent data), but every update costs energy. JouleGate charges every prediction, probe and update in joules and compares *update policies*: never, always, every *p* steps, on drift, our gradient-agreement gate (**VoU**), and published rules (CARA, UPF, D3A, replay).

- **Tasks:** time-series forecasting (PatchMixer, Granite TTM-R2, Chronos-Bolt on ETTm1, electricity, solar, synthetic) and image classification (ViT-Tiny on CORe50, PACS, Office-Home).
- **Energy:** estimated by counting every operation and memory access (priced with published 45 nm costs), plus measured CPU (RAPL) and GPU (NVML) energy.

## Quick results

Forecasting error (MASE) change vs. a frozen model, using **10 % of the energy of updating every step**:

| Stream      | Update every step | Best policy at 10 % energy |
|-------------|------------------:|---------------------------:|
| Synthetic   | −56.6 %           | **−50.6 %** (VoU)          |
| ETTm1       | +13.2 % (worse!)  | **−8.1 %** (periodic)      |
| Electricity | −7.2 %            | **−14.4 %** (periodic)     |
| Solar       | −56.0 %           | **−48.1 %** (periodic)     |

- **Updating every step is never worth it.** A tenth of the energy keeps most of the gain, and on ETTm1 always-updating is worse than never updating.
- **On real streams, a simple duty cycle wins.** *When* you update barely matters; *how many* gradient steps you spend does. This holds on three backbones, against four published rules, and on held-out seeds.
- **VoU wins when the stream has noise bursts** (synthetic): it skips updates that would chase noise.
- **Vision:** updating ViT-Tiny every step forgets 8–22 accuracy points on earlier domains; a duty cycle at ~6 % of the energy ends best or near best.
- **Hard energy budget:** if the device earns ≤ 10 % of an update's energy per step, just spend it as it arrives.

## Setup

```bash
pip install torch numpy pandas pyyaml matplotlib scipy pytest
python -m datasets.download          # datasets (public mirrors)
python -m scripts.pretrain           # PatchMixer backbone, ~15 min on CPU
python -m pytest tests/ -q
```

Released backbones also need `pip install chronos-forecasting einops`; see [`docs/details.md`](docs/details.md) for the vision data.

## Run experiments

One run, then a full sweep, then its tables and figures:

```bash
python run.py --dataset ettm1 --method vou --set method.lam=4 seed=0 tag=vou_l4   # -> results/main/
python sweep.py configs/sweeps/main.yaml                                          # 4 streams x 3 seeds: frozen, periodic, drift, VoU, oracle
python -m analysis.analyze                                                        # -> paper/figures, paper/tables
```

Each experiment is one sweep file in `configs/sweeps/` and one report script in `analysis/`:

| Experiment | Sweep(s) | Report |
|---|---|---|
| Main comparison (PatchMixer) | `main.yaml`, `gated.yaml` | `analysis.analyze` |
| Released backbones | `main_ttm.yaml`, `main_chronos.yaml` | `analysis.report_backbones` |
| Published update rules | `baselines*.yaml` | `analysis.report_backbones` |
| Measured CPU/GPU energy | `measured*.yaml`, `percall_*.yaml` | `analysis.report_measured` |
| Vision (CORe50, PACS, Office-Home) | `core50*.yaml`, `pacs*.yaml`, `officehome*.yaml` | `analysis.report_vision`, `analysis.report_paper_vision` |
| Held-out confirmation | `confirm*.yaml` (from `scripts.confirm_picks`) | `analysis.report_revision` |
| Hard energy budget | `budget*.yaml` (from `scripts.budget_sweeps`) | `analysis.report_budget` |

On a Slurm cluster, each sweep runs as an array job: `sbatch --array=0-<n-1> slurm/sweep.sbatch slurm/shards/<name>.txt` (set `JOULEGATE_ENV` to your Python env and pass `--account`).

## Repository

```
run.py, sweep.py   run one experiment / a sweep file
engine/            online loop, learners, energy meters
methods/           update policies (one file each)
models/ datasets/  backbones and data streams
configs/           base settings and sweep files
scripts/           pretraining and dev-stream setting selection
analysis/          tables and figures for the paper
```

Runs are written to `results/<experiment>/` and tables and figures to `paper/` (neither is tracked). Details on energy measurement and each setup: [`docs/details.md`](docs/details.md).
