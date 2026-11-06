# FastDriftGate: Lean specification

`FastDriftGate.lean` specifies and proves properties of the new `FastDriftGate` update in `optimizer.py`. It does not apply to the original `DriftGate` extrapolation rule.

The proof has been compiled successfully. See [VERIFIED.md](VERIFIED.md) for the toolchain and source hash.

## Check the proof

Install [elan](https://lean-lang.org/install/manual/), then:

```sh
cd speedrun/lean
lake update
lake exe cache get Mathlib/Data/Real/Sqrt.lean Mathlib/Algebra/Order/BigOperators/Ring/Finset.lean Mathlib/Tactic/Linarith.lean Mathlib/Tactic/Ring.lean Mathlib/Tactic/SplitIfs.lean
lake build
```

Lean and mathlib are pinned to v4.24.0. `lake-manifest.json` locks dependencies once resolved. The local development setup stores its toolchain in `speedrun/.elan`; to use that installation set `ELAN_HOME` to that folder when running lake. Generated toolchains and library caches are ignored by Git.

## What is specified

For a gradient coordinate `g`, the Python proposal is a blend of the bias-corrected momentum and current gradient. Lean permits an arbitrary real proposal `p`, so the property covers every blend value.

```text
a = p if g*p >= 0 else g
d = max(-cap, min(cap, a / denominator))
theta_next = (1 - lr*decay)*theta - lr*d
```

The denominator is `sqrt(v) / sqrt(1 - beta2^t) + epsilon` in the implementation. The variance recurrence starts at zero. The theorem assumes a positive bias denominator and epsilon; the variance theorem assumes beta between zero and one.

## Properties

- `variance_nonnegative`: the real-valued second-moment recurrence stays nonnegative.
- `denominator_positive`: positive bias factor and epsilon give a positive preconditioner.
- `direction_alignment`: `g*d >= 0` for each coordinate.
- `vector_alignment`: the gradient dot product with the update direction is nonnegative.
- `direction_bound`: each update coordinate has magnitude at most `cap`.
- `displacement_bound`: with no decay, each parameter changes by at most `lr*cap`.
- `displacement_bound_with_decay`: the full step changes a coordinate by at most `lr*(decay*abs(theta)+cap)`.
- `conditional_descent`: given an explicit smoothness upper bound and `L*lr*||d||² <= gradient·d`, the new loss is at most the old loss minus `(lr/2)*(gradient·d)`.

The step-size condition in the last theorem is an assumption, not a condition enforced by the optimizer. With minibatch gradients it does not imply descent of the full training objective. With weight decay, gradient alignment applies to the adaptive update component, not necessarily the entire step. Zero dot product is allowed; strict improvement is not guaranteed.

## What is not proved

No theorem establishes MNIST accuracy, runtime, generalization, global convergence, algorithmic novelty or superiority to AdamW. The formal model uses exact real arithmetic. It does not certify PyTorch, IEEE floating-point rounding, overflow, NaNs, accelerator kernels or the Python-to-Lean correspondence. Unit tests separately check the implementation against alignment and clipping properties within numerical tolerance.

There are no `sorry` placeholders or custom axioms. The printed axiom reports expose Lean’s standard foundations used by these statements.
