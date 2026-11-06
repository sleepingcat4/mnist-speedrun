# MNIST optimizer comparison

Device: cpu. Steps per run: 200. Seeds: [42, 43, 44].

| Optimizer | Test accuracy (mean) | Train seconds (mean) | Target reached |
| --- | ---: | ---: | ---: |
| adamw | 95.56% | 2.39 | 3/3 |
| driftgate | 95.64% | 2.43 | 3/3 |

A small fixed-budget experiment. No novelty, superiority, statistical significance or speed record is established.

Local validation: six unit tests passed. Reloading the saved DriftGate seed-42 model reproduced its 95.38% test accuracy.

Budget: 200 steps per optimizer per seed, batch size 128, 10,000 training images, 5,000 validation images and the official 10,000-image test set. Seeds: 42, 43, 44. CPU with four threads.

Training times exclude setup, download, warmup, validation, final test evaluation and checkpoint writes. End-to-end measured run times including validation averaged 2.87 seconds for AdamW and 2.88 seconds for DriftGate. These are measurements on this machine and will vary elsewhere.

The observed accuracy difference is +0.08 percentage points. Across-seed sample standard deviations were 0.32 percentage points for AdamW and 0.38 for DriftGate. This experiment does not establish a reliable improvement; the candidate was slightly slower on average. No post-result tuning was performed.
