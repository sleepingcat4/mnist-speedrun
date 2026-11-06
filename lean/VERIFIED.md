# Verification record

`./check.sh` completed successfully with Lean 4.24.0 and mathlib v4.24.0.

Source SHA-256: `ff358be8be7576e2ed75032adbf9dbbd9e34a078a4ff7ad2c180defd3dcce8bd`

Mathlib revision: `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`.

The axiom reports for vector alignment, the no-decay displacement bound and conditional descent contain only Lean's standard `propext`, `Classical.choice` and `Quot.sound`. There are no custom axioms or unproved placeholders in the source.

The mathematical guarantees and their assumptions are described in [README.md](README.md). This verification does not establish empirical accuracy, execution speed, stochastic convergence or floating-point implementation correctness.

Recheck from the speedrun folder:

```sh
./lean/check.sh
```
