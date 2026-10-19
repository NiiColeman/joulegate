# JouleGate: details

Setup and measurement details behind the README.

## The VoU policy and the protocol

**Value-of-update (VoU)**, the proposed policy, asks whether recent data agree on a direction to move. Each step it probes the adapter gradient `g_i` on the newest window only (a cheap pass through the adapted suffix, and its energy is charged). Since the last update it forms the unbiased estimate of the squared norm of the expected gradient,

```
U = (||sum g_i||^2 - sum ||g_i||^2) / (s (s - 1))  =  mean_{i != j} <g_i, g_j>
update iff  horizon * max(0, U)  >=  lam * measured_update_cost
```

Pure noise averages to `U ~ 0` however large the individual gradients are, while a systematic shift makes the gradients agree. `horizon * U` is a first-order prediction of the loss reduction an update buys.

Baselines: frozen (never update), periodic (every p steps; p = 1 is always-update), drift-triggered, an error-excess VoU variant (`vou_err`, ablation), a greedy hindsight oracle (not realizable), and four published update rules: CARA, a performance-forecasting rule (UPF), D3A and replay.

Protocol: policies, grids and learner settings were designed only on dev streams (separate segments or data seeds), then the evaluation streams were run with the final configs. A held-out confirmation fixes one setting per policy on the dev streams and reruns it with new seeds (3 to 5). One disclosure: an early probe of the first VoU design (now `vou_err`) on the ETTm1 and synthetic evaluation streams showed it did not beat the drift trigger, and that prompted the redesign.

## Energy measurement

`configs/energy/` selects the meter; `--energy <name>` picks one per run.

| config | what it reads | notes |
|---|---|---|
| `analytic_45nm` | MACs, elementwise ops and bytes moved per layer, priced with Horowitz (ISSCC 2014) 45 nm energies | estimated energy (a deterministic model); the default for all results |
| `measured_cpu` | RAPL package and DRAM counters (~1 ms refresh) | model on CPU; needs an exclusive node |
| `measured_gpu` | NVML energy counter of the one allocated GPU (~100 ms refresh), plus host RAPL | model on GPU (`--set device=cuda`) |
| `jetson` | a sysfs power rail, integrated | implemented, not yet run |
| `fake` | constant-power fake counters | tests and dry runs only |

The estimated-energy meter (`analytic_45nm`) is a model, not a measurement. It makes the backward-path argument explicit: backward hooks fire only for layers on the gradient path, so deeper adapters cost more backward MACs and more activation stashing.

Measured runs (`engine/measured.py`) sample the counters in a separate process, record a time window for every event, and attribute energy after the run:

1. 30 s idle, then warm-up, the stream, and 30 s idle.
2. Per event and domain: raw joules from the interpolated counters, and incremental joules = raw − idle power × duration (never clipped).
3. Categories: inference (forecasts), decision (every policy's `should_update`, including VoU's gradient probe), adaptation (updates, buffer writes, calibration), other (warm-up, logging, gaps). They must add up to the total counter energy within 1 %.

VoU's `lam` was tuned against estimated joules, and measured joules are on a different scale per device. During warm-up each measured run therefore runs full-buffer trial updates back to back for at least 2 s (one update alone is too short for the counters to resolve) and prices the same update with the estimate; VoU divides its measured prices by the ratio (`price_scale` in the summary), so `lam` keeps its meaning.

`summary.json` keeps the estimated-energy runs' field names; for measured runs `adapt_j` is incremental adaptation + decision energy, and the `energy` block holds raw and incremental totals per category and domain, idle power and its drift, and the closure error. `events.npz` has every event with its energy and, for updates, MASE over the 8 forecasts before and after and the buffer loss before and after; `samples.npz` has the raw counter series.

```bash
bash slurm/submit_measured.sh        # check job, then 4 datasets x {cpu, gpu} on exclusive nodes
python -m analysis.analyze_measured           # frontiers, budget tables, per-event energy, CPU vs GPU -> results/analysis_measured/
```

## Released forecasting backbones

Besides the reimplemented PatchMixer, the forecasting runs support two released pretrained models, selected
with `backbone` in a sweep spec or `build(..., backbone=...)`: `ttm` (Granite TTM-R2, 0.8M params, checkpoint
512-48; its model code is vendored in `models/vendor/ttm`) and `chronos` (Chronos-Bolt Tiny, 8.7M params,
`pip install chronos-forecasting einops`). Both use context 512, horizon 48 and stride 48
(`configs/base_ttm.yaml`, `configs/base_chronos.yaml`); `run.py` refuses any run whose stride is shorter than its
horizon. Weights must be cached on a machine with internet first; Slurm jobs run offline.

```bash
python -m scripts.make_retention --context 512 --horizon 48                         # retention windows for 512+48
sbatch --array=0-1 slurm/sweep.sbatch slurm/shards/dev_ttm.txt               # dev sweep (grid design)
sbatch --array=0-7 slurm/sweep.sbatch slurm/shards/eval_ttm.txt              # main + gated grids, 4 streams
sbatch slurm/measure.sbatch synthetic,ettm1,electricity,solar gpu configs/sweeps/percall_ttm.yaml results/measured_percall_ttm
python -m analysis.report_backbones                                                  # tables + figure -> paper/
```

## Vision: tiny ViT on CORe50

The same loop, policies and energy measurement on image classification under real domain shifts
(online supervised domain adaptation). A ViT-Tiny (ImageNet weights, run at 96x96) is fine-tuned on
CORe50 sessions 1 and 2 (the base domains); the stream then visits the other 9 sessions (new
backgrounds and lighting), 800 steps of 32 frames, 10 categories, and the model adapts online with
LoRA on its last 2 blocks plus the classifier head. Dev and evaluation streams use disjoint halves of
every object-session video and different session orders. Results are reported as online accuracy
(each batch is classified before its labels arrive) and retention accuracy on held-out base-domain
frames; runs store `error` = 1 - accuracy.

```bash
python -m datasets.download vit && CORE50_DOWNLOAD_ONLY=1 python -m datasets.download core50  # login node
sbatch slurm/prepare_core50.sbatch                                       # extract + convert
sbatch slurm/pretrain_vision.sbatch                                      # fine-tune on sessions 1-2
bash slurm/submit.sh core50_dev                                          # dev sweep (grid design)
python sweep.py configs/sweeps/core50.yaml --seeds 0 --workers 8         # evaluation, per seed (0, 1, 2)
EXTRA_SET="threads=8 energy.max_seconds=7200" sbatch slurm/measure.sbatch core50 cpu configs/sweeps/core50_measured.yaml results/measured_vision
sbatch slurm/measure.sbatch core50 gpu configs/sweeps/core50_percall.yaml results/measured_vision_percall
python -m analysis.report_vision                                                  # tables + figure -> paper/
```

## Scope and honest caveats

- All energy is either estimated by the model or measured on a server CPU and GPU. Nothing has run on an edge device yet; the `jetson` meter is implemented for that.
- The PatchMixer backbone is an in-repo TTM-style mixer pretrained on ETTh1, ETTh2, ETTm2, traffic (24 sensors) and exchange rate, all disjoint from the stream datasets. TTM-R2 and Chronos-Bolt are the released checkpoints.
- The synthetic stream is a constructed stress test with known regime changes and noise bursts. It checks that the mechanism behaves as designed and is not evidence on its own.
- Adaptation energy = updates + policy probes + buffer writes + one-off cost calibration. Inference is identical across policies and reported separately.
- Budget mode (`loop.budget`): absolute (`income_j`, `capacity_j`) or relative to one update's measured cost (`income_frac`, `capacity_updates`); the hard-budget experiment uses the relative form with estimated energy.
