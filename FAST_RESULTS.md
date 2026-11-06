# FastDriftGate validation and confirmation

The old implementation remains available as `DriftGate`. `FastDriftGate` is a different, gradient-aligned rule with two state buffers and coordinate clipping.

## Parameter selection

Only validation scores from seeds 101, 102 and 103 were used for selection. The same LR grid (0.001, 0.002, 0.003) was run for AdamW, DriftGate and FastDriftGate. FastDriftGate additionally tested mix values 0.05, 0.25, 0.5 and 0.75 at LR 0.003. It therefore received more tuning trials than the baselines; this is a limitation of the comparison.

| Optimizer/configuration | Validation accuracy |
| --- | ---: |
| fast-validation-001: adamw | 95.95% |
| fast-validation-001: driftgate | 95.95% |
| fast-validation-001: fastdriftgate | 95.32% |
| fast-validation-0.002: adamw | 96.66% |
| fast-validation-0.002: driftgate | 96.61% |
| fast-validation-0.002: fastdriftgate | 96.55% |
| fast-validation-0.003: adamw | 97.01% |
| fast-validation-0.003: driftgate | 97.01% |
| fast-validation-0.003: fastdriftgate | 96.96% |
| fast-validation-mix005: fastdriftgate | 93.70% |
| fast-validation-mix0.5: fastdriftgate | 97.27% |
| fast-validation-mix0.75: fastdriftgate | 97.17% |

Selected LR: 0.003 for all contenders. Selected FastDriftGate mix: 0.5; clip: 2.0. No further tuning followed the confirmation results.

## Confirmation

Five new seeds: 201–205. CPU, 4 threads, 200 updates per run, batch size 128, same CNN and fixed dataset split.

| Optimizer | Mean test accuracy | Sample std (percentage points) | Train seconds | Elapsed seconds including validation |
| --- | ---: | ---: | ---: | ---: |
| adamw | 97.22% | 0.099 | 2.538 | 3.037 |
| driftgate | 97.23% | 0.120 | 2.495 | 2.982 |
| fastdriftgate | 97.35% | 0.375 | 2.422 | 2.909 |

These are local measurements, not statistically established superiority or general performance guarantees. Timings exclude download, warmup, final test evaluation and checkpoint serialization. A small optimizer change can be hidden by model computation and system noise.

## Optimizer-only timing

Fixed-gradient CPU microbenchmark, one thread, 7 rounds of 500 updates after warmup. This tests implementation cost, not learning behavior.

- adamw: 0.1943 ms/update (median).
- driftgate: 0.4903 ms/update (median).
- fastdriftgate: 0.3518 ms/update (median).

FastDriftGate used 28.2% less optimizer-step time than DriftGate in this microbenchmark; AdamW remained faster. Buffers decrease from three to two per parameter (excluding gradients/parameters).

Raw JSON, all individual runs and trained checkpoints are under `runs/`.

## Verification

Nine Python tests pass. A saved FastDriftGate model and its two-buffer optimizer state reload successfully; reevaluation reproduces its recorded test accuracy. The Lean proof compiles with the pinned Lean 4.24.0/mathlib toolchain. See `lean/VERIFIED.md` for the source hash and scope.
