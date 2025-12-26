# SATX Full Reference

This document is a complete, user-facing reference for SATX as implemented in this repository. It describes exactly what the code does (including quirks/limitations) and provides runnable examples.

All solving examples use:

```python
satx.satisfy(solver="slime")
```

---

## 1. Overview: What SATX Is

SATX is a constraint modeling layer that compiles a restricted subset of integer arithmetic and finite-domain constraints into CNF (DIMACS) and delegates satisfiability to an external SAT solver. SATX’s “integers” are **fixed-width bit-vectors**; all variables have a finite domain determined by bit-width and signedness.

SATX is suitable for:
- Finite-domain integer constraints (bounded Diophantine problems, small algebraic identities).
- Combinatorial constraints (permutations, subset sum, packing).
- Bit-level modeling (explicit bit constraints via tensor/bit indexing).

SATX is not:
- An unbounded integer solver.
- A floating-point or real arithmetic solver.
- A general-purpose SMT solver (it only emits CNF and relies on SAT).

---

## 2. Execution Model

### 2.1 Global Engine and CNF Accumulation

SATX uses a single global engine instance (stored internally as `satx.stdlib.csp`). You must initialize it before creating variables:

```python
import satx

satx.engine(bits=10, cnf_path="tmp.cnf")
```

After `satx.engine(...)`, every operator call that involves SATX objects **immediately emits CNF clauses** into `cnf_path`. There is no separate “build” phase; the Python expression evaluation itself builds the problem.

### 2.2 Solve Boundary

Calling `satx.satisfy(solver="slime")`:
- Ensures the DIMACS header is written to the CNF file (once per engine).
- Runs the external solver on the CNF file.
- Parses `v ...` model lines from solver output.
- Decodes each SATX variable’s bits into `.value` (Python `int`), using unsigned or two’s-complement decoding depending on engine mode.
- Appends a **blocking clause** forbidding the current model (enables model enumeration by repeated calls).

### 2.3 Bit-Width Semantics

The engine’s `bits` sets the default bit-width for created integers:
- Unsigned mode (`signed=False`, default): values are interpreted in `[0, 2**bits - 1]`.
- Signed mode (`signed=True`): values are interpreted in two’s complement in `[-2**(bits-1), 2**(bits-1) - 1]`.

SATX may create auxiliary variables with different bit-widths (e.g., selection bit-vectors), and you can explicitly create variables with custom widths (e.g., `satx.integer(bits=4)`).

### 2.4 Exponent "deep" Parameter

`engine(..., deep=...)` sets the internal exponent range used when the exponent is a SATX integer (variable exponent). The implementation encodes `base ** exp` (where `exp` is a SATX integer) by selecting among `base ** i` for `i` in `1..deep` (inclusive).

### 2.5 Solver Interface (slime)

`satx.satisfy` runs a command like:

```
slime <key>.cnf <params>
```

where `<key>` is derived from `cnf_path` by stripping the file extension.

The solver must:
- Accept a DIMACS CNF path.
- Print a model using lines starting with `v ` (standard SAT solver convention).

Note: `cnf_path` must include a file extension (e.g., `.cnf`) so SATX can derive `<key>`; `satx.satisfy` raises if missing.

All examples below use `slime` as the solver name. Ensure the solver binary is on PATH (SATX does not ship a solver in this repo).

---

## 3. Core Variable Types

SATX’s primary scalar type is `satx.Unit` (a fixed-width bit-vector interpreted as an integer). The standard constructors below create `Unit` objects or collections of them.

Important: comparisons like `x < 3` **do not return a Python `bool`**; they emit constraints and return a truthy object/literal. They are meant to be used as statements (often inside `assert`) to ensure the expression is evaluated.

### 3.1 `satx.integer(bits=None)`

Purpose: create a fresh integer variable (`Unit`) using the engine’s default bit-width, or an explicit width when `bits` is provided.

Domain semantics (by engine mode):
- Unsigned engine: `[0, 2**w - 1]`
- Signed engine: `[-2**(w-1), 2**(w-1)-1]`

Minimal example:

```python
import satx

satx.engine(bits=6, cnf_path="tmp_integer.cnf")
x = satx.integer()
assert 0 < x <= 10
assert satx.satisfy(solver="slime")
assert 0 < x.value <= 10
```

Realistic use case (bounded Pythagorean constraint):

```python
import satx

satx.engine(bits=10, cnf_path="tmp_integer_use.cnf")
x = satx.integer()
y = satx.integer()
assert x**2 + y**2 == 100
assert satx.satisfy(solver="slime")
assert x.value**2 + y.value**2 == 100
```

Common misuse: using SATX expressions in Python control flow (e.g., `if x < 3:`). The condition is truthy and does not branch by value; use constraints (see Section 5).

### 3.2 `satx.constant(value, bits=None)`

Purpose: create a constant `Unit` embedded into the CNF.

Implementation detail: the `bits=` argument is ignored for constants; constants are encoded at the engine width.

Minimal example:

```python
import satx

satx.engine(bits=8, cnf_path="tmp_constant.cnf")
x = satx.integer()
two = satx.constant(2)
assert x == two + 3
assert satx.satisfy(solver="slime")
assert x.value == 5
```

Realistic use case (fixed base exponent):

```python
import satx

satx.engine(bits=16, cnf_path="tmp_constant_use.cnf")
_2 = satx.constant(2)
n = satx.integer()
x = satx.integer()
assert _2**n - 7 == x**2
assert satx.satisfy(solver="slime")
assert pow(2, n.value) - 7 == x.value**2
```

Common misuse: using negative constants in unsigned mode and expecting negative values. Unsigned decoding interprets the bit pattern in `[0, 2**bits-1]` (see Section 11).

### 3.3 `satx.vector(bits=None, size=None, is_gaussian=False, is_rational=False)`

Purpose: create a list of length `size`.

Return type:
- Default: `list[Unit]`
- `is_gaussian=True`: `list[Gaussian]` (each has `.real` / `.imag` Units)
- `is_rational=True`: `list[Rational]` (each has `.numerator` / `.denominator` Units)

Minimal example:

```python
import satx

satx.engine(bits=5, cnf_path="tmp_vector.cnf")
xs = satx.vector(size=4)
satx.apply_single(xs, lambda x: 0 <= x <= 3)
satx.all_different(xs)
assert satx.satisfy(solver="slime")
assert len(set(v.value for v in xs)) == 4
```

Realistic use case: build permutations and sequences (see `satx.permutations` in Section 12).

Common misuse: omitting `size` (it is required).

### 3.4 `satx.tensor(dimensions)`

Purpose: create a `Unit` whose bit block is also exposed through an N-dimensional nested structure (`.data`) for indexing and conditional selection.

`dimensions` defines the tensor shape; total bit count is the product of dimensions.

Bit access patterns:
- `t[i]` indexes into `.data` (may return a nested list for multi-dimensional tensors).
- `t[[i]](a, b)` returns a `Unit` equal to `a` when the selected bit is `0`, and `b` when it is `1`.

Minimal example (oddness constraint via LSB):

```python
import satx

satx.engine(bits=8, cnf_path="tmp_tensor.cnf")
x = satx.tensor(dimensions=(satx.bits(),))
assert x[[0]](0, 1) == 1
assert satx.satisfy(solver="slime")
assert x.value % 2 == 1
```

Realistic use case (subset sum by inclusion bits):

```python
import satx

universe = [3, 5, 7, 9]
target = 12
satx.engine(bits=target.bit_length(), cnf_path="tmp_tensor_subset.cnf")
pick = satx.tensor(dimensions=(len(universe),))
assert sum(pick[[i]](0, universe[i]) for i in range(len(universe))) == target
assert satx.satisfy(solver="slime")
chosen = [universe[i] for i in range(len(universe)) if pick.binary[i]]
assert sum(chosen) == target
```

Common misuse: using `t[[i]]` without calling it. `t[[i]]` returns a function; call it as `t[[i]](a, b)`.

### 3.5 `satx.index(ith, data)` and `satx.element(item, data)`

Purpose:
- `satx.index(ith, data)` returns a new `Unit` constrained to equal `data[ith]`.
- `satx.element(item, data)` returns a new `Unit` constrained to equal the index position of `item` in `data`.

`ith` may be a Python `int` or a `Unit`. `data` is a Python list of integers or `Unit`s.

Minimal example (`index`):

```python
import satx

diffs = [2, 3, 4]
satx.engine(bits=4, cnf_path="tmp_index.cnf")
v = satx.integer()
assert v == satx.index(1, diffs)
assert satx.satisfy(solver="slime")
assert v.value == diffs[1]
```

Minimal example (`element`):

```python
import satx

satx.engine(bits=4, cnf_path="tmp_element.cnf")
i = satx.element(7, [3, 5, 7, 9])
assert satx.satisfy(solver="slime")
assert i.value == 2
```

Common misuse: choosing too small an engine `bits` so indices/values cannot be represented; this can make the model UNSAT.

### 3.6 `satx.switch(x, ith, neg=False)`

Purpose: convert a bit of a `Unit` into a 0/1 `Unit` indicator.

Actual semantics (implementation): `satx.switch(x, ith)` returns:
- `1` when bit `ith` of `x` is `0`
- `0` when bit `ith` of `x` is `1`

Minimal example:

```python
import satx

satx.engine(bits=3, cnf_path="tmp_switch.cnf")
x = satx.integer(bits=3)
assert x == 0
assert satx.switch(x, 0) == 1
assert satx.satisfy(solver="slime")
assert x.value == 0
```

Common misuse: assuming it returns 1 when the bit is 1. It is the inverse indicator by design.

### 3.7 `satx.one_of(lst)`

Purpose: choose exactly one element from `lst` (SAT decision) and return the chosen value as a `Unit`.

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_one_of.cnf")
x = satx.integer()
assert x == satx.one_of([1, 3, 5])
assert satx.satisfy(solver="slime")
assert x.value in (1, 3, 5)
```

Realistic use case: modeling absolute differences as a choice between `a-b` and `b-a` (see Section 12.5).

### 3.8 `satx.oo()`

Purpose: return the maximal representable value for the current engine.

Actual semantics:
- Unsigned: `2**bits - 1`
- Signed: `2**(bits-1) - 1`

Minimal example:

```python
import satx

satx.engine(bits=5, cnf_path="tmp_oo.cnf")
assert satx.oo() == (1 << 5) - 1
```

Realistic use case: exclude sentinel max values:

```python
import satx

satx.engine(bits=6, cnf_path="tmp_oo_use.cnf")
x = satx.integer()
assert x != satx.oo()
assert satx.satisfy(solver="slime")
assert x.value != satx.oo()
```

### 3.9 `satx.Unit` (advanced)

`Unit` is the scalar type underlying SATX integers and many helper encodings.

Key properties:
- Fixed-width bit-vector with an attached engine (`.alu`).
- `.value`: decoded Python `int` after a successful `satx.satisfy(...)`; `None` before solving (or after `satx.clear`).
- `.binary`: decoded bit pattern (booleans) after solving; for tensors it is shaped by `.deep`.

Bit access and conditional selection:
- `x[i]` (single bracket, `i` int) returns the underlying bit literal (an `int` SAT literal) or a nested structure if `x` is a multi-dimensional tensor.
- `x[[i]](a, b)` (double bracket, then call) returns a `Unit` that equals `a` when bit `i` is `0` and equals `b` when bit `i` is `1`. This is the standard SATX bit-to-value idiom.
- `x.iff(bit, other)` returns a mux between `x` and `other` controlled by `bit` (a SAT literal or a special multi-bit condition when `bit` is a `Unit`).

Other user-visible helpers:
- `x.is_in(items)` constrains `x` to equal one element of `items` (internal selector encoding).
- `x.is_not_in(items)` constrains `x != item` for all `item` in `items`.
- `x.reverse(copy=True)` returns a new `Unit` with reversed bit order (used for bit-pattern constraints).

Minimal example (bit-level constraint on a scalar `Unit`):

```python
import satx

satx.engine(bits=8, cnf_path="tmp_unit_bits.cnf")
x = satx.integer()
assert x[[0]](0, 1) == 1  # odd
assert satx.satisfy(solver="slime")
assert x.value % 2 == 1
```

Realistic use case: direct bit constraints in factorization models (see Section 12.2).

### 3.10 `satx.ALU` (advanced)

`ALU` is SATX’s low-level CNF builder and bit-vector gate implementation. `satx.engine(...)` constructs an `ALU` internally and stores it in the global engine state (exported as `satx.csp`).

You typically do not instantiate `ALU` directly; `satx.satisfy(...)` operates on the global engine created by `satx.engine(...)`.

Minimal example (introspection):

```python
import satx

satx.engine(bits=4, cnf_path="tmp_alu.cnf")
assert isinstance(satx.csp, satx.ALU)
x = satx.integer()
assert x == 3
assert satx.satisfy(solver="slime")
```

Realistic use case: inspect `satx.csp.number_of_variables` / `satx.csp.number_of_clauses` after building a model to estimate CNF size.

---

## 4. Arithmetic & Logical Operations

SATX arithmetic is defined primarily by operator overloads on `satx.Unit` (and on `satx.Rational`/`satx.Gaussian` where implemented). Operators generally **emit constraints** when `.value` is `None`, and **perform concrete Python arithmetic** when `.value` is already set (after solving).

### 4.1 Addition and Subtraction: `+` / `-`

`x + y`, `x - y` create new `Unit` values and emit CNF for bit-vector addition/subtraction.

Overflow/underflow behavior depends on engine mode:
- Signed mode: addition and subtraction include explicit overflow constraints; overflow makes the problem UNSAT.
- Unsigned mode: addition constrains carry-out to 0 (overflow UNSAT). Subtraction does **not** constrain borrow/carry-out, so underflow can wrap at the bit level.

Example:

```python
import satx

satx.engine(bits=6, cnf_path="tmp_add_sub.cnf", signed=True)
x = satx.integer()
y = satx.integer()
assert x == -3
assert y == 5
z = x + y
assert z == 2
assert satx.satisfy(solver="slime")
assert z.value == 2
```

### 4.2 Multiplication: `*`

`x * y` emits a bit-vector multiplication.

Overflow behavior (implementation):
- Unsigned: overflow bits are constrained to 0 (no overflow).
- Signed: operands are sign-extended to double width, and the high half of the product is constrained to match the sign bit of the low half (range-preserving; no overflow).

### 4.3 Division: `//` (integer floor division)

`x // y` returns a `Unit` quotient constrained to match Python-style integer floor division over the decoded `.value`.
- In unsigned mode this matches the underlying unsigned long-division circuit.
- In signed mode SATX builds signed floor-division semantics using `abs(...)` + an unsigned division circuit + a remainder-based adjustment.

Division by zero is disallowed (SATX emits `assert y != 0`).

### 4.3.1 Rational Division: `/` (builds a `Rational`)

`x / y` returns a `satx.Rational(x, y)` wrapper (it does not emit a division circuit by itself). Use it when you intend to keep fractional structure (e.g., to compare two ratios).

### 4.3.2 Exact Division Helper: `satx.exact_div(x, y)`

If you need exact divisibility (remainder forced to zero), use `satx.exact_div(x, y)` and/or constrain `x % y == 0`.

### 4.4 Modulo: `%`

`x % y` returns the remainder from the underlying long-division gate.
- If `y` is 0 at the bit level, the remainder is constrained to 0.

### 4.5 Comparisons: `<, <=, >, >=, ==, !=`

Comparisons add constraints. The returned value is not a boolean; it is a SATX object/literal used only to force expression evaluation.
- Signed mode uses signed two's-complement ordering.
- Unsigned mode uses unsigned ordering.

Example:

```python
import satx

satx.engine(bits=6, cnf_path="tmp_cmp.cnf", signed=True)
x = satx.integer()
assert -5 <= x < 0
assert satx.satisfy(solver="slime")
assert -5 <= x.value < 0
```

### 4.6 Exponentiation: `**` and `pow(...)`

Supported forms:
- `base ** k` where `k` is a Python `int`: repeated multiplication.
- `base ** exp` where `exp` is a `Unit`: `exp` is constrained to be in `1..deep` (inclusive) and SATX selects among `base ** i` for `i` in that range.
- `pow(base, exp, mod)` with `base` a `Unit`: exponentiation followed by `% mod` (and an internal `assert mod != 0`).

Unsupported/partial:
- Negative exponents are not supported.
- Variable exponent forces `exp > 0` and is bounded by `deep`.

### 4.7 Absolute Value: `abs(x)`

- Signed mode: `abs(x)` returns a `Unit` selecting between `x` and `-x` based on the sign bit and also asserts `abs(x) >= 0`.
- Unsigned mode: `abs(x)` is implemented as a free SAT choice between `x` and `-x` (bit-level negation), not a mathematical absolute value over signed integers.

### 4.8 Vector Helpers: `sum`, `satx.dot`, `satx.mul`

- Python `sum([...])` works with `Unit` objects (triggers repeated `+`).
- `satx.dot(xs, ys)` is `sum(x*y for x,y in zip(xs,ys))`.
- `satx.mul(xs, ys)` is `[x*y for x,y in zip(xs,ys)]`.

### 4.9 Bitwise Operators: `& | ^`

`x & y`, `x | y`, `x ^ y` are bitwise operations on the underlying bit-vectors and return a new `Unit`.

### 4.10 Shifts: `<<` / `>>` (not bit shifts)

SATX implements:
- `x << k` as `x * (2 * k)`
- `x >> k` as `x // (2 * k)` (exact integer division)

This is not multiplication/division by `2**k` and is not a bit shift.

---

## 5. Constraint Semantics

### 5.1 `assert` as a Constraint Emitter

Most SATX constraints are written as `assert <expression>`. This is not used as a truth check; it is used to ensure the expression is evaluated so that SATX can emit CNF side effects.

Key facts:
- Many SATX operations return a truthy object/literal even when the constraint is not yet satisfied (because satisfaction is decided by the SAT solver later).
- Running Python with `-O` disables `assert` statements. This breaks models that rely on `assert` to execute constraint expressions, and it also breaks library code that contains `assert` internally (notably `Gaussian` and `Rational`).

Do not run SATX models under `python -O`.

### 5.2 Supported Python Constructs (as used by SATX)

Supported modeling patterns:
- Arithmetic expressions with `Unit` (`+`, `-`, `*`, `**`, `%`, `//`, `/`).
- Comparisons (`<`, `<=`, `==`, `!=`, `>=`, `>`), including chained comparisons (`0 < x <= 3`).
- `sum(...)`, `abs(...)`, `pow(...)` (with `Unit` base).
- Loops/list comprehensions to generate repeated constraints.

### 5.3 What Is NOT Supported (or unsafe)

- Python control flow depending on SATX expressions:
  - `if x < 3:` is not meaningful (the condition is truthy regardless of satisfiability/value).
  - `while x != 0:` is not meaningful.
- Boolean composition with `and/or/not`:
  - `or` short-circuits and can skip constraint emission.
  - `not` does not create a SATX negation constraint; it only negates Python truthiness.
  - Prefer explicit arithmetic/comparison constraints or SATX combinators (e.g., `satx.one_of`).

### 5.4 Timing: Build vs Solve vs Inspect

- **Build time:** evaluating expressions emits CNF immediately.
- **Solve time:** `satx.satisfy(...)` runs the solver and decodes `.value`.
- **Inspect time:** after solve, use `.value` / `int(x)` / `.binary` to read results.

Important nuance: after solving, `Unit` operators switch to concrete Python evaluation because `.value` is set. If you intend to reuse variables to emit new constraints after a solve, clear their `.value` first (see `satx.clear` and Section 10.4).

---

## 6. stdlib.py — Standard Constraint Library

All functions in this section are available as `satx.<name>` (because `satx.__init__` re-exports `satx.stdlib`).

For each function below:
- “Constraints” describes what is emitted into the CNF.
- Examples are minimal; larger use cases appear in Section 12.

### 6.1 System and Engine

#### `satx.version()`

What it does: prints a fixed identification banner to stdout.

Constraints: none.

Minimal example:

```python
import satx

satx.version()
```

Use case: sanity check you are running SATX (no solver involved).

#### `satx.check_engine()`

What it does: if the engine is not initialized, prints an error and exits the process.

Constraints: none.

Minimal example (normal usage is indirect):

```python
import satx

satx.engine(bits=4, cnf_path="tmp_check_engine.cnf")
satx.check_engine()
```

Use case: guard custom helper functions that assume `satx.engine(...)` was called.

#### `satx.engine(bits=None, deep=None, info=False, cnf_path="", signed=False, simplify=True)`

What it does: initializes/resets the global engine and opens `cnf_path` for CNF output.

Constraints: starts a fresh CNF; subsequent operations emit clauses immediately.

Minimal example:

```python
import satx

satx.engine(bits=8, cnf_path="tmp_engine.cnf", signed=False)
x = satx.integer()
assert x == 7
assert satx.satisfy(solver="slime")
assert x.value == 7
```

Use case: signed arithmetic (see Section 11).

When to use vs manual constraints: always required; SATX has no separate “model object”.

#### `satx.reset()`

What it does: deletes the current CNF file on disk (if it exists) and resets internal render state.

Constraints: none (but it removes the CNF file).

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_reset.cnf")
satx.reset()
```

Use case: cleanup between runs when you are managing CNF paths manually.

#### `satx.satisfy(solver, params="", log=False)`

What it does: runs the external SAT solver, decodes a model into `.value`, and appends a blocking clause.

Constraints: none new before solve; after SAT, a blocking clause is appended to the CNF file (enables model enumeration).

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_satisfy.cnf")
x = satx.integer()
assert x == 3
assert satx.satisfy(solver="slime")
assert x.value == 3
```

Use case: model enumeration or optimization loops (see Section 10).

When to use vs manual constraints: always use `satx.satisfy(solver="slime")` to solve; SATX does not include an internal SAT solver.

#### `satx.bits()`

What it does: returns the engine’s default bit-width.

Constraints: none.

Minimal example:

```python
import satx

satx.engine(bits=7, cnf_path="tmp_bits.cnf")
assert satx.bits() == 7
```

Use case: allocate tensors sized to the engine width (`satx.tensor((satx.bits(),))`).

#### `satx.oo()`

What it does: returns the maximum representable integer for the current engine.

Constraints: none.

Minimal example: see Section 3.8.

Use case: exclude sentinel max values in finite-domain models.

#### `satx.clear(lst)`

What it does: sets `.value = None` on each `Unit` in `lst`.

Constraints: none.

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_clear.cnf")
x = satx.integer()
assert x == 1
assert satx.satisfy(solver="slime")
satx.clear([x])
assert x.value is None
```

Use case: reusing variables after a solve (otherwise operations evaluate concretely); see Section 10.4.

### 6.2 Scalars and Collections

#### `satx.integer(bits=None)`

What it does: creates a fresh `Unit` variable.

Constraints: none directly; constraints arise from later operations.

Minimal example: see Section 3.1.

Use case: most models.

#### `satx.constant(value, bits=None)`

What it does: creates a constant `Unit` encoded at engine width.

Constraints: unit clauses fixing the constant’s bits.

Minimal example: see Section 3.2.

Use case: fixed bases in exponent constraints, fixed coefficients.

#### `satx.vector(bits=None, size=None, is_gaussian=False, is_rational=False)`

What it does: creates a list of variables (or `Gaussian`/`Rational` objects when requested).

Constraints: none directly.

Minimal example: see Section 3.3.

Use case: sequences, permutations, collections of decisions.

#### `satx.tensor(dimensions)`

What it does: creates a `Unit` with a shaped `.data` for bit indexing.

Constraints: none directly.

Minimal example: see Section 3.4.

Use case: bit-level modeling and conditional selection.

#### `satx.matrix(bits=None, dimensions=None, is_gaussian=False, is_rational=False)`

What it does: creates a nested Python list of variables (rows × cols).

Constraints: none directly.

Important behavior: when `is_rational=True` or `is_gaussian=True`, each row contains mixed entries (the object plus an extra scalar `Unit`) as implemented; do not assume a clean matrix of `Rational`/`Gaussian`.

Minimal example:

```python
import satx

satx.engine(bits=5, cnf_path="tmp_matrix.cnf")
m = satx.matrix(dimensions=(2, 2))
assert m[0][0] + m[1][1] == 3
assert satx.satisfy(solver="slime")
assert m[0][0].value + m[1][1].value == 3
```

Use case: Sudoku / magic squares; see Section 12.

#### `satx.reshape(lst, dimensions)`

What it does: reshapes a flat list into nested lists according to `dimensions`.

Constraints: none (pure structural helper; delegates to engine reshape).

Minimal example:

```python
import satx

satx.engine(bits=3, cnf_path="tmp_reshape.cnf")
xs = satx.vector(size=4)
grid = satx.reshape(xs, (2, 2))
assert len(grid) == 2 and len(grid[0]) == 2
```

Use case: convenience for building grids without numpy.

#### `satx.flatten(mtx)`

What it does: flattens a matrix into a list.

Constraints: none (pure Python flatten).

Minimal example:

```python
import satx

satx.engine(bits=3, cnf_path="tmp_flatten.cnf")
m = satx.matrix(dimensions=(2, 2))
flat = satx.flatten(m)
assert len(flat) == 4
```

Use case: apply constraints uniformly over a matrix (`satx.apply_single(satx.flatten(m), ...)`).

### 6.3 Subsets, Switching, and Indexing

#### `satx.subsets(lst, k=None, complement=False)`

What it does: creates a selection bit-vector and a subset view of `lst`.

Constraints: optional cardinality (`k`) constraint and conditional selection constraints.

Minimal example:

```python
import satx

universe = [3, 5, 7, 9]
target = 12
satx.engine(bits=target.bit_length(), cnf_path="tmp_subsets.cnf")
bits, subset = satx.subsets(universe)
assert sum(subset) == target
assert satx.satisfy(solver="slime")
assert sum(v.value for v in subset) == target
```

Use case: subset sum, partitioning (Section 12.1).

When to use vs manual constraints: use this when you want both a bit-vector encoding and a “masked list” representation of the chosen elements.

#### `satx.subset(lst, k, empty=None, complement=False)`

What it does: returns a masked list representing a subset with at most `k` chosen elements.
Implementation detail: due to `at_most_k`, the subset is non-empty (at least one element must be selected).

Constraints: uses an internal selection bit-vector constrained by `at_most_k`, and equates each output element to either `lst[i]` or `empty`.

Minimal example:

```python
import satx

data = [3, 5, 7, 9]
satx.engine(bits=5, cnf_path="tmp_subset.cnf")
sub = satx.subset(data, k=2)
assert sum(sub) == 12
assert satx.satisfy(solver="slime")
assert sum(v.value for v in sub) == 12
```

Use case: bounded selection (packing/cover constraints).

#### `satx.switch(x, ith, neg=False)`

What it does: returns a 0/1 `Unit` based on bit `ith` of `x` (inverse indicator; see Section 3.6).

Constraints: a bit-controlled mux between 0 and 1.

Minimal example: see Section 3.6.

Use case: bin packing, clique/vertex-cover style models.

#### `satx.one_of(lst)`

What it does: selects exactly one element of `lst` and returns it.

Constraints: internal exact-one constraint over a selector bit-vector plus conditional selection.

Minimal example: see Section 3.7.

Use case: conditional choice without Python `or`.

#### `satx.element(item, data)`

What it does: creates and returns an index variable constrained to be the position of `item` in `data`.

Constraints: one-hot index selection and consistency constraints.

Minimal example: see Section 3.5.

Use case: map “value” variables back to indices.

#### `satx.index(ith, data)`

What it does: creates and returns a value variable constrained to be `data[ith]`.

Constraints: one-hot index selection and consistency constraints.

Minimal example: see Section 3.5.

Use case: table lookup and constraints driven by Python lists.

### 6.4 Permutations and Combinatorics

#### `satx.matrix_permutation(lst, n)`

What it does: builds `(xs, ys)` where `xs` is a permutation of `0..n-1` and `ys[i]` is constrained to `lst[n * xs[i] + xs[(i + 1) % n]]` (cyclic walk over a flattened `n x n` matrix).

Constraints: permutation constraints on `xs` plus indexing constraints tying `ys` to `lst`.

Minimal example:

```python
import satx

flat = [0, 1, 1, 0]  # 2x2
satx.engine(bits=2, cnf_path="tmp_matrix_perm.cnf")
xs, ys = satx.matrix_permutation(flat, 2)
assert sum(ys) == 0
assert satx.satisfy(solver="slime")
```

Use case: TSP/Hamiltonian-style cycle cost constraints (see Section 12.7).

#### `satx.permutations(lst, n)`

What it does: entangles a length-`n` permutation of `lst`, returning `(xs, ys)` where `ys` are chosen values and `xs` are their indices in `lst`.

Constraints: element/index constraints per position, plus all-different and range constraints on `xs`.

Minimal example:

```python
import satx

items = [1, 2, 3]
satx.engine(bits=3, cnf_path="tmp_perm.cnf")
xs, ys = satx.permutations(items, len(items))
assert ys[0] + ys[1] == ys[2]
assert satx.satisfy(solver="slime")
assert sorted(v.value for v in ys) == sorted(items)
```

Use case: structured permutations and sequence puzzles.

#### `satx.combinations(lst, n)`

What it does: like `permutations`, but without the distinctness/range constraints on `xs`; repeats are allowed.

Constraints: element/index constraints per position only.

Minimal example:

```python
import satx

items = [1, 2, 3]
satx.engine(bits=3, cnf_path="tmp_comb.cnf")
xs, ys = satx.combinations(items, 2)
assert ys[0] == ys[1]
assert satx.satisfy(solver="slime")
```

Use case: selection with repetition.

#### `satx.all_binaries(lst)`

What it does: constrains every variable in `lst` to be in `{0, 1}`.

Constraints: for each element `x`, emits `0 <= x <= 1`.

Minimal example:

```python
import satx

satx.engine(bits=2, cnf_path="tmp_all_binaries.cnf")
xs = satx.vector(size=3)
satx.all_binaries(xs)
assert satx.satisfy(solver="slime")
assert all(v.value in (0, 1) for v in xs)
```

Use case: binary linear systems (Section 12.8).

### 6.5 Constraint Combinators

#### `satx.apply_single(lst, f, indexed=False)`

What it does: calls `f` on each element of `lst` to emit constraints.
- `indexed=False`: calls `f(x)`
- `indexed=True`: calls `f(i, x)`

Constraints: whatever `f` emits.

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_apply_single.cnf")
xs = satx.vector(size=4)
satx.apply_single(xs, lambda x: 0 <= x <= 3)
assert satx.satisfy(solver="slime")
```

Use case: domain bounding and bulk constraints.

#### `satx.apply_dual(lst, f, indexed=False)`

What it does: calls `f` on each unordered pair `(i, j)` with `i < j`.

Constraints: whatever `f` emits.

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_apply_dual.cnf")
xs = satx.vector(size=4)
satx.apply_dual(xs, lambda a, b: a != b)
assert satx.satisfy(solver="slime")
```

Use case: all-different, diagonal constraints (N-Queens), pairwise ordering.

#### `satx.apply_different(lst, f, indexed=False)`

What it does: calls `f` on each ordered pair `(i, j)` with `i != j`.

Constraints: whatever `f` emits.

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_apply_different.cnf")
xs = satx.vector(size=3)
satx.apply_different(xs, lambda a, b: a != b)
assert satx.satisfy(solver="slime")
```

Use case: directed constraints where both directions matter.

#### `satx.all_different(args)`

What it does: constrains all pairs in `args` to be different.

Constraints: emits `x != y` for all `i < j`.

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_all_different.cnf")
xs = satx.vector(size=4)
satx.all_different(xs)
assert satx.satisfy(solver="slime")
assert len(set(v.value for v in xs)) == 4
```

Use case: permutations, Sudoku, Latin squares/cubes.

#### `satx.all_out(args, values)`

What it does: constrains each `x` in `args` to be different from every value in `values`.

Constraints: emits `x != v` for each `x` and each `v`.

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_all_out.cnf")
xs = satx.vector(size=3)
satx.all_out(xs, values=[0, 1])
assert satx.satisfy(solver="slime")
assert all(v.value not in (0, 1) for v in xs)
```

Use case: forbid labels/sentinels.

#### `satx.all_in(args, values)`

What it does: constrains each `x` in `args` to be equal to one of `values` (via `one_of`).

Constraints: per element, emits `x == one_of(values)`.

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_all_in.cnf")
xs = satx.vector(size=3)
satx.all_in(xs, values=[2, 4, 6])
assert satx.satisfy(solver="slime")
assert all(v.value in (2, 4, 6) for v in xs)
```

Use case: discrete-domain restriction without manual disjunctions.

#### `satx.at_most_k(x, k)`

What it does: constrains a bit-vector `x` so that no subset of `k+1` bits can all be 0. This is used in internal encodings where “selected” is represented by a 0 bit.
Implementation detail: the current encoding also enforces at least one 0 bit (non-empty selection). `k=0` yields UNSAT.

Constraints: a combinational (subset) CNF encoding over the bits in `x.block`.

Minimal example:

```python
import satx

satx.engine(bits=6, cnf_path="tmp_at_most_k.cnf")
sel = satx.integer(bits=5)
satx.at_most_k(sel, k=2)
assert satx.satisfy(solver="slime")
```

Use case: internal building block for `subset`, `element`, and similar helpers.

### 6.6 Arithmetic Helpers

#### `satx.dot(xs, ys)`

What it does: returns `sum(x*y for x,y in zip(xs,ys))`.

Constraints: those emitted by the underlying `*` and `+` operations.

Minimal example:

```python
import satx

satx.engine(bits=6, cnf_path="tmp_dot.cnf")
x = satx.vector(size=2)
w = [1, 2]
assert satx.dot(x, w) == 5
assert satx.satisfy(solver="slime")
assert x[0].value + 2 * x[1].value == 5
```

Use case: linear constraints (Section 12.8).

#### `satx.mul(xs, ys)`

What it does: elementwise multiplication: `[x*y for x,y in zip(xs,ys)]`.

Constraints: those emitted by each `*`.

Minimal example:

```python
import satx

satx.engine(bits=6, cnf_path="tmp_mul.cnf")
x = satx.vector(size=2)
y = satx.vector(size=2)
z = satx.mul(x, y)
assert z[0] + z[1] == 5
assert satx.satisfy(solver="slime")
assert z[0].value + z[1].value == 5
```

Use case: vectorized arithmetic in modeling.

#### `satx.values(lst, cleaner=None)`

What it does: returns `[x.value for x in lst]`, optionally filtered by `cleaner`.

Constraints: none.

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_values.cnf")
xs = satx.vector(size=3)
satx.all_binaries(xs)
assert satx.satisfy(solver="slime")
vals = satx.values(xs)
assert all(v in (0, 1) for v in vals)
```

Use case: extracting solutions for validation or post-processing.

#### `satx.factorial(x)`

What it does: returns a `Unit` encoding of `x!` using internal selection and repeated multiplication.

Constraints: internal selection constraints plus arithmetic constraints.

Minimal example:

```python
import satx
import math

satx.engine(bits=32, cnf_path="tmp_factorial.cnf")
x = satx.integer()
assert satx.factorial(x) == math.factorial(6)
assert satx.satisfy(solver="slime")
assert math.factorial(x.value) == math.factorial(6)
```

Use case: symbolic inversion of factorial for small ranges (expensive).

#### `satx.sigma(f, i, n)`

What it does: returns an encoding of the sum `f(i) + f(i+1) + ... + f(n)` (with `i` a Python int, `n` a SATX `Unit`).

Constraints: internal selection tying `n` to a chosen endpoint plus arithmetic constraints.

Minimal example:

```python
import satx

satx.engine(bits=16, cnf_path="tmp_sigma.cnf")
x = satx.integer()
n = satx.integer()
assert satx.sigma(lambda k: k**2, 1, n) == x
assert satx.satisfy(solver="slime")
assert x.value == sum(k**2 for k in range(1, n.value + 1))
```

Use case: encoded sums of sequences (expensive).

#### `satx.pi(f, i, n)`

What it does: returns an encoding of the product `f(i) * f(i+1) * ... * f(n)`.

Constraints: internal selection plus repeated multiplication constraints.

Minimal example:

```python
import satx
import functools
import operator

satx.engine(bits=32, cnf_path="tmp_pi.cnf")
x = satx.integer()
n = satx.integer()
assert satx.pi(lambda k: k**2, 1, n) == x
assert n > 0
assert satx.satisfy(solver="slime")
prod = functools.reduce(operator.mul, (k**2 for k in range(1, n.value + 1)), 1)
assert x.value == prod
```

Use case: encoded products (very expensive).

#### `satx.sqrt(x)`

What it does: constrains `x` to be a perfect square and returns `y` such that `x == y**2`.

Constraints: equality constraint `x == y**2`.

Minimal example:

```python
import satx

satx.engine(bits=10, cnf_path="tmp_sqrt.cnf")
x = satx.integer()
y = satx.sqrt(x)
assert y == 6
assert satx.satisfy(solver="slime")
assert x.value == 36
```

Use case: norms and Pythagorean constraints.

#### `satx.rotate(x, k)`

What it does: returns a new `Unit` whose bits are a rotation of `x` by `k` positions (mod engine width).

Constraints: bitwise equality constraints between rotated positions.

Minimal example:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_rotate.cnf")
x = satx.integer()
assert x == 1
y = satx.rotate(x, 1)
assert satx.satisfy(solver="slime")
assert y.value == 2
```

Use case: bit-level encodings.

#### `satx.is_prime(p)` / `satx.is_not_prime(p)`

What it does: adds a Fermat-like constraint using `pow(2, p, p)`.

Constraints: `is_prime(p)` emits `pow(2, p, p) == 2`. `is_not_prime(p)` emits `p != 2` and `pow(2, p, p) != 2`.

Minimal example:

```python
import satx

satx.engine(bits=6, cnf_path="tmp_prime.cnf")
p = satx.integer()
assert p == 5
satx.is_prime(p)
assert satx.satisfy(solver="slime")
assert p.value == 5
```

Minimal example (`is_not_prime`):

```python
import satx

satx.engine(bits=6, cnf_path="tmp_not_prime.cnf")
p = satx.integer()
assert p == 4
satx.is_not_prime(p)
assert satx.satisfy(solver="slime")
assert p.value == 4
```

Use case: bounded filtering; not a complete primality test (see Section 13).

### 6.7 Gaussian/Rational Constructors

#### `satx.gaussian(x=None, y=None)`

What it does: returns a `Gaussian(x, y)` object. If no args, it creates fresh `Unit` numerator/denominator variables via `satx.integer()`.

Constraints: if created from scratch, none beyond those later added by Gaussian operations.

Minimal example:

```python
import satx

satx.engine(bits=8, cnf_path="tmp_gaussian_ctor.cnf", signed=True)
g = satx.gaussian()
assert g.real == 3
assert g.imag == -4
assert satx.satisfy(solver="slime")
assert int(g.real) == 3 and int(g.imag) == -4
```

Use case: Gaussian integer arithmetic (Section 8).

#### `satx.rational(x=None, y=None)`

What it does: returns a `Rational(x, y)` object. If no args, it creates fresh numerator/denominator `Unit`s via `satx.integer()`.

Constraints: when denominator is a `Unit`, construction asserts denominator is not zero.

Minimal example:

```python
import satx
from fractions import Fraction

satx.engine(bits=8, cnf_path="tmp_rational_ctor.cnf", signed=True)
r = satx.rational()
assert r == Fraction(1, 2)
assert satx.satisfy(solver="slime")
assert Fraction(int(r.numerator), int(r.denominator)) == Fraction(1, 2)
```

---

## 7. gcc.py — (Document Actual Purpose, Not Assumed)

`satx.gcc` is a small set of “global constraint catalog”-style macros implemented on top of SATX primitives. These helpers **emit SATX constraints** (they are not separate solvers).

Import:

```python
import satx
import satx.gcc as gcc
```

### 7.1 `gcc.abs_val(x, y)`

What it does (implementation): emits `y >= 0` and `abs(x) == y`.

Constraints: comparison constraints and the SATX `abs` encoding.

Minimal example (use signed mode for meaningful abs):

```python
import satx
import satx.gcc as gcc

satx.engine(bits=6, cnf_path="tmp_gcc_abs.cnf", signed=True)
x = satx.integer()
y = satx.integer()
gcc.abs_val(x, y)
assert x == -3
assert satx.satisfy(solver="slime")
assert y.value == 3
```

Use case: absolute deviation constraints in small optimization loops.

### 7.2 `gcc.all_differ_from_at_least_k_pos(k, lst)`

What it does: for each pair of distinct vectors in `lst`, enforces they differ in at least `k` positions.

Constraints: builds two “nil” sentinel values (not equal to any entry) and uses a tensor-based selector encoding; constrains a per-position difference indicator sum to be `>= k`.

Minimal example (2 vectors of length 3):

```python
import satx
import satx.gcc as gcc

satx.engine(bits=4, cnf_path="tmp_gcc_atleast.cnf")
v1 = satx.vector(size=3)
v2 = satx.vector(size=3)
gcc.all_differ_from_at_least_k_pos(2, [v1, v2])
assert satx.satisfy(solver="slime")
diff = sum(int(v1[i]) != int(v2[i]) for i in range(3))
assert diff >= 2
```

Use case: Hamming-distance style constraints.

### 7.3 `gcc.all_differ_from_at_most_k_pos(k, lst)`

What it does: for each pair of distinct vectors in `lst`, enforces they differ in at most `k` positions.

Constraints: similar tensor-based selector encoding to the “at least” form, but constrains equality indicators to be `>= len(vec) - k`.

Minimal example: see below (includes both `all_differ_from_at_most_k_pos` and `all_differ_from_exactly_k_pos`).

### 7.4 `gcc.all_differ_from_exactly_k_pos(k, lst)`

What it does: enforces “exactly k” differences by calling both `all_differ_from_at_least_k_pos` and `all_differ_from_at_most_k_pos`.

Minimal example (`at_most_k_pos` and `exactly_k_pos`):

```python
import satx
import satx.gcc as gcc

satx.engine(bits=4, cnf_path="tmp_gcc_atmost_exactly.cnf")
v1 = satx.vector(size=3)
v2 = satx.vector(size=3)

gcc.all_differ_from_at_most_k_pos(1, [v1, v2])
assert satx.satisfy(solver="slime")
diff = sum(v1[i].value != v2[i].value for i in range(3))
assert diff <= 1
```

```python
import satx
import satx.gcc as gcc

satx.engine(bits=4, cnf_path="tmp_gcc_exactly.cnf")
v1 = satx.vector(size=3)
v2 = satx.vector(size=3)

gcc.all_differ_from_exactly_k_pos(1, [v1, v2])
assert satx.satisfy(solver="slime")
diff = sum(v1[i].value != v2[i].value for i in range(3))
assert diff == 1
```

### 7.5 `gcc.all_equal(lst)`

What it does: constrains all variables in `lst` to be equal.

Constraints: emits `x == y` for all pairs via `satx.apply_dual`.

Minimal example:

```python
import satx
import satx.gcc as gcc

satx.engine(bits=5, cnf_path="tmp_gcc_all_equal.cnf")
xs = satx.vector(size=3)
gcc.all_equal(xs)
assert xs[0] == 4
assert satx.satisfy(solver="slime")
assert all(v.value == 4 for v in xs)
```

Use case: symmetry reduction / tying variables together.

### 7.6 `gcc.all_different(lst)`

What it does: delegates to `satx.all_different(lst)`.

### 7.7 `gcc.element(idx, lst, val)`

What it does: emits `val == satx.index(idx, lst)`.

Minimal example:

```python
import satx
import satx.gcc as gcc

satx.engine(bits=6, cnf_path="tmp_gcc_element.cnf")
idx = satx.integer()
val = satx.integer()
gcc.element(idx, [10, 11, 12], val)
assert idx == 1
assert satx.satisfy(solver="slime")
assert val.value == 11
```

Use case: catalog-style macro wrapper around `satx.index`.

### 7.8 `gcc.gcd(x, y, z)`

What it does (implementation): emits a specific gcd-like constraint pattern:
- coerces `y` to a SATX constant if it is not a `Unit`
- constrains `0 < x <= y`, `z > 0`
- constrains `z == y % x`
- constrains `satx.exact_div(x, z) % (y % z) == 0` (exact division helper)

This is not a complete Euclidean-algorithm encoding; it is exactly the constraints above.

Minimal example (structural):

```python
import satx
import satx.gcc as gcc

satx.engine(bits=6, cnf_path="tmp_gcc_gcd.cnf")
x = satx.integer()
z = satx.integer()
gcc.gcd(x, 10, z)
assert x == 3
assert satx.satisfy(solver="slime")
```

Use case: experimental number-theoretic modeling; validate externally after solve.

### 7.9 `gcc.sort(lst1, lst2)`

What it does: constrains `lst2` to be a sorted permutation of `lst1`.

Constraints (implementation):
- builds a permutation `(xs, ys) = satx.permutations(lst1, len(lst1))`
- constrains each `lst2[i] == ys[i]`
- constrains `lst2` to be non-decreasing via pairwise `<=`

Minimal example:

```python
import satx
import satx.gcc as gcc

satx.engine(bits=4, cnf_path="tmp_gcc_sort.cnf")
lst1 = [3, 1, 2]
lst2 = satx.vector(size=3)
gcc.sort(lst1, lst2)
assert satx.satisfy(solver="slime")
assert [v.value for v in lst2] == sorted(lst1)
```

Use case: canonicalization for symmetry breaking.

### 7.10 `gcc.sort_permutation(lst_from, lst_per, lst_to)`

What it does: constrains `lst_to` to be a sorted permutation of `lst_from`, and constrains `lst_per` to be the corresponding index permutation.

Constraints (implementation):
- `lst_to` is constrained non-decreasing
- a fresh permutation `(xs1, ys1) = satx.permutations(lst_from, len(lst_from))`
- uses Python list equality (`assert ys1 == lst_to`, `assert lst_per == xs1`) to emit elementwise equality constraints

Minimal example:

```python
import satx
import satx.gcc as gcc

satx.engine(bits=4, cnf_path="tmp_gcc_sort_perm.cnf")
lst_from = [3, 1, 2]
lst_to = satx.vector(size=3)
lst_per = satx.vector(size=3)
gcc.sort_permutation(lst_from, lst_per, lst_to)
assert satx.satisfy(solver="slime")
assert [v.value for v in lst_to] == sorted(lst_from)
assert sorted(v.value for v in lst_per) == list(range(3))
```

Use case: keep both the sorted values and the permutation indices.

### 7.11 `gcc.count(val, lst, rel, lim)`

What it does: counts occurrences of `val` in `lst` and relates the count to `lim` using `rel`.

Constraints (implementation):
- creates a tensor `t` of length `len(lst)`
- for each position, emits `t[[i]](0, lst[i] - val) == 0` (equality indicator encoding)
- emits `rel(sum(t[[i]](0,1) for i), lim)`

Minimal example (at least two ones):

```python
import satx
import satx.gcc as gcc

satx.engine(bits=4, cnf_path="tmp_gcc_count.cnf")
xs = satx.vector(size=4)
satx.all_binaries(xs)
gcc.count(1, xs, rel=lambda a, b: a >= b, lim=2)
assert satx.satisfy(solver="slime")
assert sum(v.value for v in xs) >= 2
```

Use case: global cardinality constraints.

---

## 8. gaussian.py — Gaussian / Linear Algebra Utilities

`satx.gaussian` defines a Gaussian-integer wrapper type (`satx.Gaussian`). It does not implement Gaussian elimination; “linear algebra” in SATX is modeled by writing linear constraints directly (typically using `satx.dot`, `satx.matrix`, and `assert`).

### 8.1 `satx.Gaussian`

Representation: `Gaussian(real, imag)` where `real` and `imag` are typically `satx.Unit` objects.

Public attributes:
- `g.real`: real part (`Unit`)
- `g.imag`: imaginary part (`Unit`)

Supported operations (as implemented):
- `g1 == g2`: emits constraints `g1.real == g2.real` and `g1.imag == g2.imag` (uses `assert` internally).
- `g1 != g2`: emits a constraint that at least one part differs (uses `assert` internally).
- unary `-`, binary `+`, `-`, `*`.
- `/`: Gaussian division implemented via scalar arithmetic; component divisions produce `Rational` values (so divisibility is not required, but denominators are constrained non-zero).
- `g ** k` for Python `int` `k >= 1`: repeated multiplication.
- `abs(g)`: returns `Gaussian(sqrt(real**2 + imag**2), 0)` (a Gaussian whose imaginary part is 0).
- `g.conjugate()`: returns `Gaussian(real, -imag)`.
- `complex(g)`: converts to a Python complex using `int(real)` and `int(imag)` (requires a solved model).

Important constraint note: `Gaussian.__eq__` and `Gaussian.__ne__` use internal `assert` statements to emit constraints. Running with `python -O` disables those asserts and breaks Gaussian constraints.

Minimal example:

```python
import satx

satx.engine(bits=8, cnf_path="tmp_gaussian_min.cnf", signed=True)
g = satx.gaussian()
assert g.real == 3
assert g.imag == -4
assert satx.satisfy(solver="slime")
assert int(g.real) == 3 and int(g.imag) == -4
```

Realistic use case (Gaussian factorization over a small target):

```python
import satx

satx.engine(bits=8, cnf_path="tmp_gaussian_mul.cnf", signed=True)
a = satx.gaussian()
b = satx.gaussian()
target = satx.gaussian(satx.constant(5), satx.constant(0))
assert a * b == target
assert satx.satisfy(solver="slime")
got = complex(int(a.real), int(a.imag)) * complex(int(b.real), int(b.imag))
assert got == 5 + 0j
```

### 8.2 Solving a Linear System (modeled directly)

SATX does not provide a built-in elimination routine. You model linear systems by writing constraints.

Example: solve the integer system `2x + 3y = 7` and `x - y = 1`:

```python
import satx

satx.engine(bits=6, cnf_path="tmp_linear.cnf", signed=True)
x = satx.integer()
y = satx.integer()
assert 2 * x + 3 * y == 7
assert x - y == 1
assert satx.satisfy(solver="slime")
assert 2 * x.value + 3 * y.value == 7
assert x.value - y.value == 1
```

---

## 9. rational.py — Rational Arithmetic & Constraints

SATX rationals are represented as `(numerator, denominator)` pairs, typically both `satx.Unit` objects, wrapped by `satx.Rational`.

### 9.1 `satx.Rational`

Representation: `Rational(numerator, denominator)`.

Public attributes:
- `r.numerator`: numerator (`Unit` or Python int)
- `r.denominator`: denominator (`Unit` or Python int)

Constructor constraint (implementation):
- If `denominator` is a `Unit`, construction emits `assert denominator != 0`.

Supported operations (as implemented):
- Equality `==`:
  - If compared to a `Unit`, emits constraints equivalent to `numerator == denominator * other`.
  - If compared to any object with `.numerator` and `.denominator` (including Python `int` and `fractions.Fraction`), emits constraints enforcing proportionality (a shared scaling factor).
- Inequality `!=`: emits the cross-multiplication constraint `denominator * other.numerator != numerator * other.denominator` (requires the other operand to have `.numerator`/`.denominator`).
- Arithmetic: unary `-`, `+`, `-`, `*`, `/`, and `**` for Python `int` powers.
- Comparisons: `<=, >=, <, >` against `Unit` (treated as integer) or against `.numerator`/`.denominator` objects.

Important limitations/quirks:
- `Rational` does not implement `%`. `Rational.__pow__(..., modulo=...)` attempts `other % modulo` and will raise if `modulo` is provided.
- Some methods (`__repr__`, `__float__`, `__abs__`, `invert`) contain `if self.denominator == 0:` guards. If `denominator` is an *unsolved* `Unit`, `denominator == 0` emits a constraint and is truthy; avoid calling these methods during model construction. Use them after solving, or avoid them entirely in constraint-building code.
- Mixed arithmetic with `Unit` is not symmetric: `r * unit` is supported (special-cased in `Rational.__mul__`), but `unit * r` is not supported by `Unit.__mul__`.

Minimal example (constrain to a fraction):

```python
import satx
from fractions import Fraction

satx.engine(bits=8, cnf_path="tmp_rational_min.cnf", signed=True)
r = satx.rational()
assert r == Fraction(1, 2)
assert satx.satisfy(solver="slime")
assert Fraction(int(r.numerator), int(r.denominator)) == Fraction(1, 2)
```

Realistic use case (rational equation):

```python
import satx
from fractions import Fraction

satx.engine(bits=10, cnf_path="tmp_rational_eq.cnf", signed=True)
x = satx.rational()
y = satx.rational()
assert x**3 + x * y == y**2
assert x != 0
assert y != 0
assert satx.satisfy(solver="slime")
xf = Fraction(int(x.numerator), int(x.denominator))
yf = Fraction(int(y.numerator), int(y.denominator))
assert xf != 0 and yf != 0
assert xf**3 + xf * yf == yf**2
```

### 9.2 Notes on Integer vs Rational Behavior

- Rationals inherit signed/unsigned behavior from the underlying `Unit`s in numerator/denominator; use `signed=True` if you need negative rational values decoded as negatives.
- SATX does not normalize rationals; many numerator/denominator pairs can represent the same value.

---

## 10. Solving & Result Inspection

### 10.1 `satx.satisfy` Return Value

`satx.satisfy(solver="slime")` returns:
- `True` when a SAT model was found and parsed from solver output.
- `False` when no model was found (UNSAT or no parsable model lines).

On `False`, `Unit.value` remains `None` for variables.

### 10.2 Reading Values

After `satx.satisfy(...)` returns `True`:
- Scalars: read `x.value` or `int(x)`.
- Bits: read `x.binary` (booleans) for `Unit`s and tensors.
- Rationals: read `int(r.numerator)` / `int(r.denominator)`; build `fractions.Fraction` if needed.
- Gaussians: read `int(g.real)` / `int(g.imag)`; build `complex(...)` if needed.

Example:

```python
import satx

satx.engine(bits=6, cnf_path="tmp_read.cnf", signed=True)
x = satx.integer()
assert x == -3
assert satx.satisfy(solver="slime")
assert x.value == -3
assert int(x) == -3
```

### 10.3 Enumerating Multiple Models

SATX appends a blocking clause for each SAT model it finds. Calling `satx.satisfy(...)` repeatedly (without adding new constraints) enumerates distinct models until UNSAT.

```python
import satx

satx.engine(bits=2, cnf_path="tmp_enum.cnf")
x = satx.integer()
satx.all_binaries([x])
seen = []
while satx.satisfy(solver="slime"):
    seen.append(x.value)
assert set(seen) == {0, 1}
```

### 10.4 Re-solving After a SAT Call (clearing values)

After a SAT call, `Unit.value` is set and many `Unit` operators switch to concrete Python arithmetic/comparison. If you intend to emit *new constraints* using previously solved variables, clear them first:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_reuse.cnf")
x = satx.integer()
assert 0 <= x <= 3
assert satx.satisfy(solver="slime")

satx.clear([x])
assert x == 2
assert satx.satisfy(solver="slime")
assert x.value == 2
```

If you do not clear, `assert x == 2` becomes a Python boolean check and does not add CNF.

---

## 11. Negative Numbers (Explicit Section)

### 11.1 Enabling Signed Integers

Enable negative-number semantics with:

```python
import satx

satx.engine(bits=8, cnf_path="tmp_signed.cnf", signed=True)
```

SATX uses **two's complement** decoding in `satx.satisfy` when `signed=True`:
- The most-significant bit is the sign bit.
- Range is `[-2**(bits-1), 2**(bits-1)-1]`.

### 11.2 Constants in Unsigned vs Signed Mode

Negative constants are encoded as bit patterns. In unsigned mode, the decoded value is in `[0, 2**bits-1]`:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_neg_unsigned.cnf", signed=False)
x = satx.integer()
assert x == -1
assert satx.satisfy(solver="slime")
assert x.value == 15
```

In signed mode, the same bit pattern decodes to the negative value:

```python
import satx

satx.engine(bits=4, cnf_path="tmp_neg_signed.cnf", signed=True)
x = satx.integer()
assert x == -1
assert satx.satisfy(solver="slime")
assert x.value == -1
```

### 11.3 Overflow Behavior (signed)

In signed mode, SATX forbids overflow for:
- `+` and `-` (explicit signed overflow constraints).
- `*` (product must fit in width; high bits constrained to sign extension).

Overflow results in UNSAT instead of wrap-around.

Example (overflow UNSAT):

```python
import satx

satx.engine(bits=4, cnf_path="tmp_overflow.cnf", signed=True)
x = satx.integer()
y = satx.integer()
assert x == 7
assert y == 1
assert x + y == -8  # wrap-around would be -8; overflow is forbidden
assert not satx.satisfy(solver="slime")
```

### 11.4 Signed Comparisons and `abs`

Signed comparisons (`<`, `<=`, `>`, `>=`) use signed two's-complement ordering when `signed=True`.

`abs(x)` in signed mode returns a non-negative `Unit` and additionally emits `assert abs(x) >= 0`.

Example (negative values):

```python
import satx

satx.engine(bits=6, cnf_path="tmp_abs_signed.cnf", signed=True)
x = satx.integer()
y = satx.integer()
assert x < 0
assert y < 0
assert abs(x - y) == 1
assert satx.satisfy(solver="slime")
assert abs(x.value - y.value) == 1
```

### 11.5 Division/Modulo with Negatives (limitation)

Exact division uses an unsigned long-division gate under the hood. Signed `//` and `%` are built on top of that (via `abs(...)` + adjustment) to match Python floor-division/modulo semantics.

---

## 12. Advanced Mathematical Use Cases

Each example below is a complete snippet: it builds constraints, solves with `slime`, and validates the solution by re-evaluating the intended math over `.value`.

### 12.1 Subset Sum

```python
import satx

universe = [3, 5, 7, 9]
target = 12

satx.engine(bits=target.bit_length(), cnf_path="tmp_case_subset_sum.cnf")
bits, subset = satx.subsets(universe)
assert sum(subset) == target

assert satx.satisfy(solver="slime")
subset_values = [v.value for v in subset]
assert sum(subset_values) == target
assert all(v in (0, universe[i]) for i, v in enumerate(subset_values))
```

### 12.2 RSA-Style Factorization (small)

```python
import satx

rsa = 3007
satx.engine(bits=rsa.bit_length(), cnf_path="tmp_case_rsa.cnf")
p = satx.integer()
q = satx.integer()
assert p * q == rsa

assert satx.satisfy(solver="slime")
assert p.value * q.value == rsa
```

### 12.3 Difference of Squares

```python
import satx

rsa = 3007
satx.engine(bits=rsa.bit_length() + 1, cnf_path="tmp_case_diff_squares.cnf")
p = satx.integer()
q = satx.integer()
assert p**2 - q**2 == rsa
assert q < p

assert satx.satisfy(solver="slime")
assert p.value**2 - q.value**2 == rsa
assert q.value < p.value
```

### 12.4 Exponential Diophantine Equation

```python
import satx

satx.engine(bits=32, cnf_path="tmp_case_exp_dioph.cnf")
_2 = satx.constant(2)
_3 = satx.constant(3)
x = satx.integer()
y = satx.integer()
assert _3**x == y * _2**x + 1

assert satx.satisfy(solver="slime")
assert pow(3, x.value) == y.value * pow(2, x.value) + 1
```

### 12.5 Linear System (integer)

```python
import satx

satx.engine(bits=6, cnf_path="tmp_case_linear.cnf", signed=True)
x = satx.integer()
y = satx.integer()
assert 2 * x + 3 * y == 7
assert x - y == 1

assert satx.satisfy(solver="slime")
assert 2 * x.value + 3 * y.value == 7
assert x.value - y.value == 1
```

### 12.6 Mixed Integer/Rational Constraints

```python
import satx
from fractions import Fraction

satx.engine(bits=10, cnf_path="tmp_case_mix.cnf", signed=True)
x = satx.integer()
r = satx.rational()
assert x == 3
assert r == Fraction(1, 2)

assert satx.satisfy(solver="slime")
assert x.value == 3
assert Fraction(int(r.numerator), int(r.denominator)) == Fraction(1, 2)
```
## 13. Limitations & Design Constraints

- Global mutable state: SATX uses a single global engine; it is not thread-safe and not re-entrant.
- Fixed-width integers: all values are bit-vectors; you must choose bit-widths that bound the intended solution space.
- Overflow handling:
  - Addition and multiplication are encoded as overflow-free at the chosen width; overflow makes the model UNSAT.
  - Unsigned subtraction does not constrain borrow/carry-out and can wrap at the bit level.
- Integer division/modulo: `//` and `%` emit constraints and disallow division by zero (`assert y != 0`).
- Shifts are not bit shifts: `<<`/`>>` are implemented as multiply/divide by `2*k`, not by `2**k`.
- Exact divisibility: use `satx.exact_div(x, y)` and/or `assert x % y == 0`.
- Constraints depend on `assert`: many library components emit constraints via internal `assert` statements (`Gaussian`, `Rational`). `python -O` disables them and breaks modeling.
- `matrix(is_rational=True|is_gaussian=True)` returns mixed rows (object plus an extra scalar per column) as implemented; do not assume a clean matrix of objects.
- `is_prime`/`is_not_prime` use a Fermat-like check (`pow(2, p, p)`); they are not complete primality/compositeness characterizations.
- Performance characteristics: CNF size grows quickly with non-linear operations (multiplication, variable exponentiation, factorial/sigma/pi) and with quadratic constraints (`all_different` over large sets).

---

## 14. Best Practices

- Always call `satx.engine(..., cnf_path="something.cnf")` before creating variables.
- Keep bit-widths small and explicit; enlarge only when necessary.
- Prefer `signed=True` when modeling with negatives or signed comparisons.
- Avoid Python control flow over SATX expressions (`if`, `while`); use constraints instead.
- Avoid `and/or/not` to combine constraints; write separate `assert` statements or use `satx.one_of`.
- After a SAT call, either rebuild a fresh engine or call `satx.clear([...])` before reusing variables to emit new constraints.
- Validate models by recomputing the intended math on `.value` after solve.
- For UNSAT debugging, reduce `bits`, simplify constraints, and isolate sub-constraints to find the contradiction source.

---

## 15. Complete Public API Reference

This section enumerates all public callables exposed by:
- `satx` (re-exports of `satx.stdlib` plus `Unit`, `ALU`, `Gaussian`, `Rational`)
- `satx.gcc`

### 15.1 `satx` (alphabetical)

- `ALU(bits=None, deep=None, cnf="")` — CNF/bit-vector engine class (`satx/alu.py`).
- `Gaussian(x, y)` — Gaussian integer wrapper class (`satx/gaussian.py`).
- `Rational(x, y)` — Rational wrapper class (`satx/rational.py`).
- `Unit(alu, key=None, block=None, value=None, bits=None, deep=None)` — Scalar bit-vector integer class (`satx/unit.py`).
- `all_binaries(lst)` — Constrain each element to `{0,1}` (`satx/stdlib.py`).
- `all_different(args)` — Pairwise inequality constraint (`satx/stdlib.py`).
- `all_in(args, values)` — Restrict each element to a discrete set (`satx/stdlib.py`).
- `all_out(args, values)` — Exclude a set of forbidden values (`satx/stdlib.py`).
- `apply_different(lst, f, indexed=False)` — Apply `f` to all ordered pairs `i != j` (`satx/stdlib.py`).
- `apply_dual(lst, f, indexed=False)` — Apply `f` to all unordered pairs `i < j` (`satx/stdlib.py`).
- `apply_single(lst, f, indexed=False)` — Apply `f` to each element (`satx/stdlib.py`).
- `at_most_k(x, k)` — Cardinality-style constraint on bit-vector zeros (`satx/stdlib.py`).
- `bits()` — Return engine bit-width (`satx/stdlib.py`).
- `check_engine()` — Exit if engine not initialized (`satx/stdlib.py`).
- `clear(lst)` — Clear `.value` on each `Unit` (`satx/stdlib.py`).
- `combinations(lst, n)` — Element selection with repetition allowed (`satx/stdlib.py`).
- `constant(value, bits=None)` — Constant `Unit` (bits arg ignored) (`satx/stdlib.py`).
- `dot(xs, ys)` — Dot product helper (`satx/stdlib.py`).
- `element(item, data)` — Constrain/return index of `item` in `data` (`satx/stdlib.py`).
- `engine(bits=None, deep=None, info=False, cnf_path="", signed=False, simplify=True)` — Initialize/reset engine (`satx/stdlib.py`).
- `factorial(x)` — Factorial encoding (`satx/stdlib.py`).
- `flatten(mtx)` — Flatten a matrix into a list (`satx/stdlib.py`).
- `gaussian(x=None, y=None)` — Gaussian constructor helper (`satx/stdlib.py`).
- `hess_abstract(xs, oracle, f, g, log=None, fast=False, cycles=1, target=0)` — HESS optimizer (abstract) (`satx/stdlib.py`).
- `hess_binary(n, oracle, fast=False, cycles=1, target=0, seq=None)` — HESS optimizer (binary) (`satx/stdlib.py`).
- `hess_sequence(n, oracle, fast=False, cycles=1, target=0, seq=None)` — HESS optimizer (sequence) (`satx/stdlib.py`).
- `hyper_loop(n, m)` — Nested-loop index generator (`satx/stdlib.py`).
- `index(ith, data)` — Constrain/return `data[ith]` (`satx/stdlib.py`).
- `integer(bits=None)` — Fresh integer variable (`satx/stdlib.py`).
- `is_not_prime(p)` — Fermat-like non-prime constraint (`satx/stdlib.py`).
- `is_prime(p)` — Fermat-like prime constraint (`satx/stdlib.py`).
- `matrix(bits=None, dimensions=None, is_gaussian=False, is_rational=False)` — Matrix constructor (`satx/stdlib.py`).
- `matrix_permutation(lst, n)` — Cycle through flattened matrix by permutation (`satx/stdlib.py`).
- `mul(xs, ys)` — Elementwise multiply helper (`satx/stdlib.py`).
- `one_of(lst)` — Exact-one choice combinator (`satx/stdlib.py`).
- `oo()` — Maximum representable value for current engine (`satx/stdlib.py`).
- `permutations(lst, n)` — Permutation entanglement (`satx/stdlib.py`).
- `pi(f, i, n)` — Product encoding (`satx/stdlib.py`).
- `rational(x=None, y=None)` — Rational constructor helper (`satx/stdlib.py`).
- `reset()` — Delete CNF file and reset render state (`satx/stdlib.py`).
- `reshape(lst, dimensions)` — Reshape list to nested structure (`satx/stdlib.py`).
- `rotate(x, k)` — Bit rotation constraint (`satx/stdlib.py`).
- `satisfy(solver, params="", log=False)` — Run solver and decode model (`satx/stdlib.py`).
- `sigma(f, i, n)` — Sum encoding (`satx/stdlib.py`).
- `sqrt(x)` — Perfect-square encoding (`satx/stdlib.py`).
- `subset(lst, k, empty=None, complement=False)` — Subset-of-at-most-k structure (non-empty selection) (`satx/stdlib.py`).
- `subsets(lst, k=None, complement=False)` — Selection bits + masked subset (`satx/stdlib.py`).
- `switch(x, ith, neg=False)` — Bit-to-indicator helper (`satx/stdlib.py`).
- `tensor(dimensions)` — Tensor/bit-array `Unit` (`satx/stdlib.py`).
- `values(lst, cleaner=None)` — Extract `.value` list (`satx/stdlib.py`).
- `vector(bits=None, size=None, is_gaussian=False, is_rational=False)` — Vector constructor (`satx/stdlib.py`).
- `version()` — Print system info banner (`satx/stdlib.py`).

### 15.2 `satx.gcc` (documented subset, alphabetical)

Note: `satx.gcc` exposes additional helpers beyond this list; see `docs/gcc_coverage.md` and `satx/gcc.py` for the full set.

- `abs_val(x, y)` — Constrain `y >= 0` and `abs(x) == y` (`satx/gcc.py`).
- `all_differ_from_at_least_k_pos(k, lst)` — Pairwise differences in at least k positions (`satx/gcc.py`).
- `all_differ_from_at_most_k_pos(k, lst)` — Pairwise differences in at most k positions (`satx/gcc.py`).
- `all_differ_from_exactly_k_pos(k, lst)` — Pairwise differences in exactly k positions (`satx/gcc.py`).
- `all_different(lst)` — Alias to `satx.all_different` (`satx/gcc.py`).
- `all_equal(lst)` — Constrain all variables equal (`satx/gcc.py`).
- `count(val, lst, rel, lim)` — Count occurrences and apply relation (`satx/gcc.py`).
- `element(idx, lst, val)` — `val == satx.index(idx, lst)` (`satx/gcc.py`).
- `gcd(x, y, z)` — GCD-like constraint pattern (`satx/gcc.py`).
- `sort(lst1, lst2)` — Sorted permutation constraint (`satx/gcc.py`).
- `sort_permutation(lst_from, lst_per, lst_to)` — Sorted permutation + permutation indices (`satx/gcc.py`).

### 15.3 Coverage Audit

Enumerated public symbol sets:
- `satx` exposes 53 callables (49 stdlib functions + `Unit`, `ALU`, `Gaussian`, `Rational`): all listed in Section 15.1 and documented in Sections 2-14.
- `satx.gcc` exposes 38 public helpers (excluding underscore-prefixed internals); Section 15.2 documents a subset. See `docs/gcc_coverage.md` and `satx/gcc.py` for the full list.

Use case: rational constraints (Section 9).

### 6.8 Non-SAT Utilities

#### `satx.hyper_loop(n, m)`

What it does: yields all length-`n` vectors over `range(m)` in a nested-loop order.

Constraints: none.

Minimal example:

```python
import satx

assert list(satx.hyper_loop(2, 2)) == [[0, 0], [0, 1], [1, 0], [1, 1]]
```

Use case: iterate over index grids when building constraints.

#### `satx.hess_sequence(n, oracle, fast=False, cycles=1, target=0, seq=None)`

What it does: heuristic search over permutations/sequences to minimize `oracle(seq)`.

Constraints: none (pure Python).

Minimal example:

```python
import satx

def oracle(seq):
    return sum(seq)

best = satx.hess_sequence(n=5, oracle=oracle, fast=True, cycles=1)
assert sorted(best) == list(range(5))
```

Use case: heuristic pre-search before SATX modeling.

#### `satx.hess_binary(n, oracle, fast=False, cycles=1, target=0, seq=None)`

What it does: heuristic search over boolean vectors to minimize `oracle(bits)`.

Constraints: none.

Minimal example:

```python
import satx

def oracle(bits):
    return sum(bits)

best = satx.hess_binary(n=8, oracle=oracle, fast=True, cycles=1)
assert sum(best) <= 8
```

Use case: heuristic configuration search.

#### `satx.hess_abstract(xs, oracle, f, g, log=None, fast=False, cycles=1, target=0)`

What it does: heuristic search over an abstract state `xs` using user-provided move operators `f` and `g`.

Constraints: none.

Minimal example:

```python
import satx

xs = bytearray([0, 1, 2, 3])

def oracle(s):
    return sum(s)

def f(i, j, s):
    s[i], s[j] = s[j], s[i]

def g(i, j, s):
    s[i], s[j] = s[j], s[i]

best = satx.hess_abstract(xs, oracle, f, g, fast=True, cycles=1)
assert isinstance(best, (list, bytearray))
```

Use case: heuristic search over custom encodings.

---
