import Mathlib.Data.Real.Sqrt
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.SplitIfs

/-!
Real-arithmetic specification of the *update transform* in FastDriftGate.
The momentum proposal and positive preconditioner are inputs. This does not verify
PyTorch floating-point execution or prove stochastic-training convergence.
-/
open scoped BigOperators

namespace FastDriftGate

/-- Variance recurrence from a zero initialization. -/
noncomputable def variance (beta : ℝ) (g : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => beta * variance beta g t + (1 - beta) * (g t)^2

theorem variance_nonnegative (beta : ℝ) (g : ℕ → ℝ)
    (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1) (t : ℕ) :
    0 ≤ variance beta g t := by
  induction t with
  | zero => simp [variance]
  | succ t ih =>
    exact add_nonneg (mul_nonneg hb0 ih)
      (mul_nonneg (sub_nonneg.mpr hb1) (sq_nonneg _))

theorem denominator_positive (v bias epsilon : ℝ)
    (hb : 0 < bias) (he : 0 < epsilon) :
    0 < Real.sqrt v / bias + epsilon := by
  have h := div_nonneg (Real.sqrt_nonneg v) (le_of_lt hb)
  linarith

noncomputable def align (g proposal : ℝ) : ℝ :=
  if 0 ≤ g * proposal then proposal else g

noncomputable def clip (cap value : ℝ) : ℝ :=
  max (-cap) (min cap value)

noncomputable def direction (g proposal denominator cap : ℝ) : ℝ :=
  clip cap (align g proposal / denominator)

theorem align_nonnegative_product (g proposal : ℝ) :
    0 ≤ g * align g proposal := by
  unfold align
  split_ifs with h
  · exact h
  · exact mul_self_nonneg g

theorem clip_bounds (cap value : ℝ) (hcap : 0 ≤ cap) :
    -cap ≤ clip cap value ∧ clip cap value ≤ cap := by
  unfold clip
  constructor
  · exact le_max_left _ _
  · exact max_le (by linarith) (min_le_left _ _)

theorem clip_nonneg (cap value : ℝ) (hcap : 0 ≤ cap) (hv : 0 ≤ value) :
    0 ≤ clip cap value := by
  exact le_max_of_le_right (le_min hcap hv)

theorem clip_nonpos (cap value : ℝ) (hcap : 0 ≤ cap) (hv : value ≤ 0) :
    clip cap value ≤ 0 := by
  exact max_le (neg_nonpos.mpr hcap) (le_trans (min_le_right _ _) hv)

/-- Each update coordinate is aligned with the current gradient coordinate. -/
theorem direction_alignment (g proposal denominator cap : ℝ)
    (hd : 0 < denominator) (hc : 0 ≤ cap) :
    0 ≤ g * direction g proposal denominator cap := by
  have ha := align_nonnegative_product g proposal
  unfold direction
  rcases lt_trichotomy g 0 with hg | hg | hg
  · have hs : align g proposal ≤ 0 := by
      rcases mul_nonneg_iff.mp ha with h | h
      · linarith [h.1]
      · exact h.2
    exact mul_nonneg_of_nonpos_of_nonpos (le_of_lt hg)
      (clip_nonpos cap _ hc (div_nonpos_of_nonpos_of_nonneg hs (le_of_lt hd)))
  · simp [hg]
  · have hs : 0 ≤ align g proposal := by
      rcases mul_nonneg_iff.mp ha with h | h
      · exact h.2
      · linarith [h.1]
    exact mul_nonneg (le_of_lt hg) (clip_nonneg cap _ hc (div_nonneg hs (le_of_lt hd)))

/-- Coordinate clipping bounds update size independently of the proposal. -/
theorem direction_bound (g proposal denominator cap : ℝ) (hc : 0 ≤ cap) :
    |direction g proposal denominator cap| ≤ cap := by
  exact abs_le.mpr (clip_bounds cap _ hc)

/-- Summing the coordinate guarantee yields nonnegative gradient dot update. -/
theorem vector_alignment {n : ℕ} (g proposal denominator : Fin n → ℝ) (cap : ℝ)
    (hd : ∀ i, 0 < denominator i) (hc : 0 ≤ cap) :
    0 ≤ ∑ i, g i * direction (g i) (proposal i) (denominator i) cap := by
  exact Finset.sum_nonneg (fun i _ => direction_alignment _ _ _ _ (hd i) hc)

/-- Without weight decay, the absolute coordinate displacement is at most lr*cap. -/
theorem displacement_bound (theta g proposal denominator cap lr : ℝ)
    (hc : 0 ≤ cap) (hlr : 0 ≤ lr) :
    |(theta - lr * direction g proposal denominator cap) - theta| ≤ lr * cap := by
  have h := direction_bound g proposal denominator cap hc
  have heq : (theta - lr * direction g proposal denominator cap) - theta =
      -(lr * direction g proposal denominator cap) := by ring
  rw [heq, abs_neg, abs_mul, abs_of_nonneg hlr]
  exact mul_le_mul_of_nonneg_left h hlr

/-- The full decoupled-weight-decay update also has an explicit displacement bound. -/
theorem displacement_bound_with_decay (theta g proposal denominator cap lr decay : ℝ)
    (hc : 0 ≤ cap) (hlr : 0 ≤ lr) (hw : 0 ≤ decay) :
    |((1 - lr * decay) * theta - lr * direction g proposal denominator cap) - theta|
      ≤ lr * (decay * |theta| + cap) := by
  have heq : ((1 - lr * decay) * theta - lr * direction g proposal denominator cap) - theta =
      -(lr * (decay * theta + direction g proposal denominator cap)) := by ring
  rw [heq, abs_neg, abs_mul, abs_of_nonneg hlr]
  apply mul_le_mul_of_nonneg_left _ hlr
  calc
    |decay * theta + direction g proposal denominator cap|
      ≤ |decay * theta| + |direction g proposal denominator cap| := abs_add_le _ _
    _ ≤ decay * |theta| + cap := by
      rw [abs_mul, abs_of_nonneg hw]
      exact add_le_add_left (direction_bound _ _ _ _ hc) _

/-- A conditional descent result using an explicit smoothness upper bound.
`dot` is gradient·direction and `normsq` is squared direction norm.
The step restriction must be checked; alignment alone is not a convergence proof.
This statement excludes decoupled weight decay and assumes an exact full gradient. -/
theorem conditional_descent (oldLoss newLoss lr L dot normsq : ℝ)
    (hlr : 0 ≤ lr) (hdot : 0 ≤ dot)
    (hsmooth : newLoss ≤ oldLoss - lr * dot + (L / 2) * lr^2 * normsq)
    (hstep : L * lr * normsq ≤ dot) :
    newLoss ≤ oldLoss - (lr / 2) * dot ∧ newLoss ≤ oldLoss := by
  have hscaled := mul_le_mul_of_nonneg_left hstep hlr
  have hdescent : newLoss ≤ oldLoss - (lr / 2) * dot := by nlinarith
  constructor
  · exact hdescent
  · have hnonneg := mul_nonneg hlr hdot
    nlinarith

end FastDriftGate

#print axioms FastDriftGate.vector_alignment
#print axioms FastDriftGate.displacement_bound
#print axioms FastDriftGate.conditional_descent
