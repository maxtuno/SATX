# SATX Roadmap (Fixed-Point)

- Add mixed-scale normalization helper (LCM rescale) so `Fixed` addition can be explicit and safe when scales differ.
- Reuse shared product gates in fixed-point multiply-rescale to reduce CNF size across repeated products.
- Provide `Unit`-side comparisons with `Fixed` (explicit helpers only, no implicit casting) to avoid asymmetric operator support.
- Add `fixed_bounds` helper to propagate min/max bounds for early UNSAT pruning and scale/bits guidance.
- Add `fixed_advice(..., explain=True)` optional text explanations to complement numeric output.
- Add deterministic enumeration helper that respects `Fixed` variables and scale in blocking clauses.
