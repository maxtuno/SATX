Fixed-point Release Notes (SATX)

Changelog (short)
- Added `satx.fixed_advice(...)` for numeric scale/bit guidance (no constraints, no output).
- `satx.clear(...)` now accepts `Fixed` and clears underlying `.raw`.
- `Fixed` auto-promotes `int` and `Unit` operands when used from the `Fixed` side.
- `Fixed.__str__` prints clean decimals for power-of-10 scales, fractions otherwise.
- Added overflow-risk warnings in `Fixed.__mul__`/`__pow__` (UserWarning; one per scale/bits).
- Added `satx.as_int(...)` with explicit rounding policies (`exact|floor|ceil|round`).
- Added `fixed_mul_floor` / `fixed_mul_round` helpers for explicit rounding in multiply-rescale.
- Added `engine(..., max_fixed_pow=...)` guard for Fixed exponentiation.
- Added doc smoke tests for fixed-point snippets and fixed-default mode.
- Expanded test coverage + deterministic 200-case fuzz smoke test for fixed-point ops.

Possible improvements (prioritized)
1) Add a scale-normalization helper for mixed-scale addition (explicit LCM rescale).
2) Optimize multiply rescale by reusing shared product gates across expressions (CNF reduction).
3) Add `Fixed`-aware comparisons on the `Unit` side (explicitly documented; no implicit casting).
4) Add a lightweight `fixed_bounds` helper to propagate max/min bounds and reduce UNSAT cases early.
5) Add optional `fixed_advice(..., explain=True)` to include a short text explanation (opt-in).
6) Add a deterministic model enumeration helper that respects `Fixed` variables.
