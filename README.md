# MNIST Speedrun

I built this project to test how quickly a very small CNN can learn MNIST under a fixed training budget. It compares AdamW with two experimental optimizers: DriftGate and FastDriftGate.

Every optimizer receives the same model, initial weights, minibatches and number of updates. The script saves the trained models and produces a Markdown comparison automatically.

## Results

Five confirmation runs were completed with 200 updates per optimizer on CPU.

| Optimizer | Mean test accuracy | Training time | Optimizer step time |
| --- | ---: | ---: | ---: |
| AdamW | 97.22% | 2.538 s | 0.1943 ms |
| DriftGate | 97.23% | 2.495 s | 0.4903 ms |
| FastDriftGate | **97.35%** | **2.422 s** | 0.3518 ms |

FastDriftGate used 28.2% less optimizer-step time than DriftGate in the isolated benchmark. AdamW still had the cheapest optimizer update. The accuracy differences are small and these results do not prove general superiority.

The complete tuning history, timing definitions and limitations are in [FAST_RESULTS.md](FAST_RESULTS.md).

## Run it

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/sleepingcat4/mnist-speedrun.git
cd mnist-speedrun
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

MNIST is downloaded and checksum-verified on the first run. The default benchmark uses 10,000 training images, 200 updates, a batch size of 128 and three seeds. CUDA, MPS and CPU are supported.

Run a quick smoke test:

```bash
python train.py --device cpu --steps 20 --eval-every 10 --seeds 42 --output runs/smoke
```

Reproduce the five-seed comparison:

```bash
python train.py \
  --device cpu \
  --threads 4 \
  --steps 200 \
  --seeds 201 202 203 204 205 \
  --output runs/confirmation
```

Run only AdamW and FastDriftGate:

```bash
python train.py --optimizers adamw fastdriftgate
```

## FastDriftGate

FastDriftGate keeps exponential moving averages of the gradient and squared gradient. It mixes the corrected momentum with the current gradient, rejects directions that disagree with the current gradient, normalizes by the second moment and clips each coordinate.

```text
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient²
proposal = (1 - mix) * m_hat + mix * gradient
aligned = proposal if gradient * proposal >= 0 else gradient
update = clamp(aligned / (sqrt(v_hat) + epsilon), -clip, clip)
parameter = (1 - learning_rate * weight_decay) * parameter - learning_rate * update
```

It uses two state tensors per parameter instead of DriftGate's three. The defaults selected using validation data are a learning rate of `0.003`, mix of `0.5` and clipping value of `2.0`.

This is an experimental combination of known optimization ideas. I am not claiming that it is globally novel or that it will outperform AdamW on other models and datasets.

## Fair comparison

- MNIST's training set is divided into a fixed training partition and 5,000-image validation partition.
- The official test set is used after training, not for tuning.
- Initial weights and batch indices match for every optimizer within a seed.
- Model structure, normalization, batch size and weight decay are identical.
- Warmup, downloads, final evaluation and checkpoint writing are excluded from training time.
- Optimizer order rotates across seeds to reduce consistent ordering bias.

Use `--validation-only` when tuning parameters so the test set remains untouched.

## Tests

```bash
python -m unittest discover -s tests -v
python benchmark_optimizer.py
```

The nine tests cover update alignment, clipping, argument validation, missing and zero gradients, sparse-gradient rejection, paired batch order and checkpoint restoration.

## Lean proof

The checked Lean proof covers nonnegative variance, gradient alignment, coordinate update bounds and a conditional smoothness-based descent result.

```bash
cd lean
lake build
```

The proof uses Lean 4.24.0 with a pinned mathlib revision. It proves exact real-number properties of the update rule. It does not certify PyTorch floating-point execution, global convergence, runtime, accuracy or novelty. See [lean/README.md](lean/README.md) and [lean/VERIFIED.md](lean/VERIFIED.md) for the exact scope.

## Output

Each run saves:

```text
runs/experiment/
├── config.json
├── summary.json
├── comparison.md
└── optimizer-seed/
    ├── final.pt
    └── metrics.json
```

Use a new `--output` directory for each experiment. Only load PyTorch checkpoints that you trust.

## Author

Built by Ammar ([@sleepingcat4](https://github.com/sleepingcat4)).
