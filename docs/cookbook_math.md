# SATX Mathematical Cookbook (by problem class)

## Front matter

This cookbook is a problem-driven set of SATX modeling patterns (CNF compilation + SAT solving) with short, runnable examples you can copy/paste and adapt. It only covers functionality that exists in SATX as implemented in this repository.

### Template (copy/paste)

```python
import satx

satx.engine(bits=8, cnf_path="tmp_template.cnf", signed=False)

x = satx.integer()
assert 0 <= x <= 10

assert satx.satisfy(solver='slime')
assert 0 <= x.value <= 10

satx.reset()
```

## Recipe Index

- [A) Boolean & CNF-ish modeling patterns (bridge layer)](#a-boolean--cnf-ish-modeling-patterns-bridge-layer)
- [B) Integer feasibility (Diophantine – linear)](#b-integer-feasibility-diophantine--linear)
- [C) Integer feasibility (Diophantine – non-linear)](#c-integer-feasibility-diophantine--non-linear)
- [D) Exponential / power constraints](#d-exponential--power-constraints)
- [E) Modular / congruence-like constraints](#e-modular--congruence-like-constraints)
- [F) Rational constraints (from rational.py)](#f-rational-constraints-from-rationalpy)
- [G) Linear algebra / Gaussian helpers (from gaussian.py)](#g-linear-algebra--gaussian-helpers-from-gaussianpy)
- [H) gcc.py domain recipes](#h-gccpy-domain-recipes)
- [I) Optimization patterns (iterative bounding)](#i-optimization-patterns-iterative-bounding)
- [Appendix](#appendix)

## A) Boolean & CNF-ish modeling patterns (bridge layer)

### A1. Bit reification and inverse switches (`Unit[[i]](...)`, `satx.switch`)

**Problem statement.** Use a 1-bit decision to select between two values, and read the decision as a 0/1 variable.

**Modeling pattern (SATX).**
- Create a 1-bit selector: `sel = satx.integer(bits=1)`.
- Use `sel[[0]](a, b)` as a mux: returns `a` when the bit is 0, `b` when the bit is 1.
- `satx.switch(sel, 0)` is an *inverse* indicator: returns 1 when the bit is 0, 0 when the bit is 1.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_a1.cnf")

sel = satx.integer(bits=1)
x = satx.integer()

assert x == sel[[0]](3, 7)
assert x > 5

inv = satx.switch(sel, 0)     # 1 when sel bit is 0, else 0
direct = sel[[0]](0, 1)       # 0 when sel bit is 0, else 1

assert satx.satisfy(solver='slime')

assert x.value == 7
assert sel.value == 1
assert direct.value == 1
assert inv.value == 0
```

**Scaling notes / pitfalls.**
- `satx.switch` is inverted (1 means the bit is 0). If you need the direct bit value, prefer `sel[[0]](0, 1)` (or a 1-bit `Unit` itself).
- Piecewise modeling builds both branches; keep branches small.

**Variants.**
- Invert a switch: `satx.switch(x, i, neg=True)`.
- If you don’t need an explicit selector, use `satx.one_of([a, b])`.

### A2. Exactly-one choice among values (`satx.one_of`)

**Problem statement.** Constrain a variable to take exactly one value from a finite set.

**Modeling pattern (SATX).**
- Use `satx.one_of([v1, v2, ...])` to create an internal exact-one selector and return the chosen value.

**Minimal code.**

```python
import satx

satx.engine(bits=4, cnf_path="tmp_a2.cnf")

x = satx.integer()
assert x == satx.one_of([1, 3, 5])
assert x != 3

assert satx.satisfy(solver='slime')

assert x.value in (1, 5)
```

**Scaling notes / pitfalls.**
- `one_of` introduces an internal selector of length `len(list)`; CNF grows with list length and value bit-width.
- Don’t write `a or b` expecting a SAT disjunction; Python short-circuits (see Section 5 in the full reference).

**Variants.**
- Constrain a whole vector into a discrete set: `satx.all_in(xs, values=[...])`.
- Choose among *expressions*: `assert y == satx.one_of([x + 1, 2 * x, 3 * x])`.

### A3. Table lookup and inverse lookup (`satx.index`, `satx.element`)

**Problem statement.** (1) Select a value from a Python list by an index variable. (2) Find the index of a given item.

**Modeling pattern (SATX).**
- `satx.index(ith, data)` creates a fresh variable constrained to `data[ith]`.
- `satx.element(item, data)` creates a fresh variable constrained to the index position of `item` in `data`.
- If `ith` is a SATX integer, add range constraints: `0 <= ith < len(data)`.

**Minimal code.**

```python
import satx

data = [2, 5, 7, 11]

satx.engine(bits=6, cnf_path="tmp_a3.cnf")

ith = satx.integer()
assert 0 <= ith < len(data)

v = satx.index(ith, data)
assert v == 7

pos = satx.element(11, data)

assert satx.satisfy(solver='slime')

assert v.value == 7
assert data[ith.value] == 7
assert pos.value == data.index(11)
```

**Scaling notes / pitfalls.**
- `index/element` are selector encodings; large tables increase CNF size.
- Use explicit range bounds for index variables; otherwise the selector constraints may force UNSAT unexpectedly.

**Variants.**
- Index into a list of SATX integers (not just Python ints).
- Use `gcc.element(idx, lst, val)` (Section H) as a catalog-style wrapper macro.

### A4. Piecewise constraints (explicit mux)

**Problem statement.** Define `y` as one of two expressions depending on a 1-bit selector.

**Modeling pattern (SATX).**
- Build both branches (`e0`, `e1`) and mux them with `sel[[0]](e0, e1)`.

**Minimal code.**

```python
import satx

satx.engine(bits=6, cnf_path="tmp_a4.cnf")

sel = satx.integer(bits=1)
x = satx.integer()
y = satx.integer()

e0 = x + 1
e1 = 2 * x

assert x == 2
assert y == sel[[0]](e0, e1)
assert y == 4

assert satx.satisfy(solver='slime')

assert y.value == 4
assert (x.value + 1 == y.value) or (2 * x.value == y.value)
```

**Scaling notes / pitfalls.**
- Both branches exist in CNF even if one is “selected”; keep expressions small.
- Avoid Python `if/else` on SATX expressions; it won’t branch by satisfiability.

**Variants.**
- Use `satx.one_of([e0, e1])` if you don’t need the selector value.
- Use tensors for multi-way piecewise (`t[[i]](a, b)` per bit, then combine).

### A5. Direct CNF modeling (SAT bridge with 1-bit tensors)

**Problem statement.** Encode a small CNF instance directly and let SATX act as a bridge to the external SAT solver.

**Modeling pattern (SATX).**
- Use `satx.engine(bits=1, ...)` so each variable is 1-bit (boolean).
- Use `x = satx.tensor(dimensions=(n,))` to create `n` boolean variables.
- A literal `ℓ` becomes `x[[var]](ℓ < 0, ℓ > 0)` (truth value of the literal).
- Combine with bitwise `|` and `&` (not Python `or/and`).

**Minimal code.**

```python
import functools
import operator
import satx

clauses = [[1, -2], [2, 3], [-1, -3]]  # 1-based variable ids, signed literals
n = 3

satx.engine(bits=1, cnf_path="tmp_a5.cnf")
x = satx.tensor(dimensions=(n,))

formula = functools.reduce(
    operator.iand,
    (
        functools.reduce(
            operator.ior,
            (x[[abs(lit) - 1]](lit < 0, lit > 0) for lit in cls),
        )
        for cls in clauses
    ),
)

assert formula == 1
assert satx.satisfy(solver='slime')

assignment = list(map(bool, x.binary))
for cls in clauses:
    assert any((assignment[abs(lit) - 1] if lit > 0 else not assignment[abs(lit) - 1]) for lit in cls)
```

**Scaling notes / pitfalls.**
- This is “raw SAT”; for arithmetic constraints, prefer SATX’s integer operators.
- Use `|`/`&` on 1-bit units; Python `or/and` short-circuit and can skip constraints.

**Variants.**
- Add a cardinality constraint over booleans using `satx.switch`/sums (see Section E).

## B) Integer feasibility (Diophantine – linear)

### B1. Simple bounded inequalities (range + linear sum)

**Problem statement.** Find integers `x, y` in small ranges such that `x + y > 2`.

**Modeling pattern (SATX).**
- Use chained comparisons (`0 < x <= 3`) to emit multiple constraints.
- Use standard arithmetic (`+`) and comparisons.

**Minimal code.**

```python
import satx

satx.engine(bits=10, cnf_path="tmp_b1.cnf")
x = satx.integer()
y = satx.integer()

assert 0 < x <= 3
assert 0 < y <= 3
assert x + y > 2

assert satx.satisfy(solver='slime')
assert 0 < x.value <= 3
assert 0 < y.value <= 3
assert x.value + y.value > 2
```

**Scaling notes / pitfalls.**
- Chained comparisons are fine, but do not use SATX expressions in Python `if`/`while`.
- Pick `bits` large enough to represent all constants and intermediate sums.

**Variants.**
- Apply the same bounds to many variables with `satx.apply_single([x1, x2, ...], lambda v: lo <= v <= hi)`.

### B2. Linear systems with dot products (`satx.dot`, `satx.vector`)

**Problem statement.** Find integer weights `w0, w1` and bias `b` that match a small linear dataset.

**Modeling pattern (SATX).**
- Use `satx.dot(x_row, w)` for linear forms.
- Constrain weights/bias to small domains with `apply_single`.

**Minimal code.**

```python
import satx

x_data = [(0, 0), (0, 1), (1, 0), (1, 1)]
y_data = [sum(row) for row in x_data]  # 0,1,1,2

satx.engine(bits=4, cnf_path="tmp_b2.cnf")
w = satx.vector(size=2)
b = satx.integer()

satx.apply_single(w + [b], lambda v: 0 <= v <= 2)

for row, y in zip(x_data, y_data):
    assert satx.dot(list(row), w) + b == y

assert satx.satisfy(solver='slime')

wv = [v.value for v in w]
assert all(0 <= v <= 2 for v in wv)
assert all(sum(row[j] * wv[j] for j in range(2)) + b.value == y for row, y in zip(x_data, y_data))
```

**Scaling notes / pitfalls.**
- Keep weights bounded; otherwise SAT may waste time exploring irrelevant high values.
- `satx.dot` multiplies and sums; bit-width must cover worst-case sum.

**Variants.**
- Use `satx.matrix(dimensions=(n, m))` for multiple weight vectors; apply constraints row-wise.

### B3. Bounded integer programming with binaries (`satx.all_binaries`)

**Problem statement.** Select items (0/1) so that weight ≤ capacity and total value matches a target.

**Modeling pattern (SATX).**
- Use `satx.all_binaries(xs)` for 0/1 decision variables.
- Use linear constraints on weighted sums.

**Minimal code.**

```python
import satx

weights = [3, 4, 5]
values = [4, 5, 6]
capacity = 7
target_value = 9  # must pick items 0 and 1

satx.engine(bits=6, cnf_path="tmp_b3.cnf")
pick = satx.vector(size=len(weights))
satx.all_binaries(pick)

assert sum(weights[i] * pick[i] for i in range(len(weights))) <= capacity
assert sum(values[i] * pick[i] for i in range(len(values))) == target_value

assert satx.satisfy(solver='slime')

pv = [v.value for v in pick]
assert all(v in (0, 1) for v in pv)
assert sum(weights[i] * pv[i] for i in range(len(weights))) <= capacity
assert sum(values[i] * pv[i] for i in range(len(values))) == target_value
```

**Scaling notes / pitfalls.**
- Prefer `bits` just large enough for the largest weighted sum.
- For larger problems, CNF grows roughly with (#items × bit-width) plus arithmetic gates.

**Variants.**
- Replace `all_binaries` with bit tensors (`satx.tensor`) if you want direct bit access.

### B4. Subset sum via masked subsets (`satx.subsets`)

**Problem statement.** Choose a subset of numbers that sums to a target.

**Modeling pattern (SATX).**
- `bits, subset = satx.subsets(universe)` returns a masked subset list (each entry is 0 or the item).
- Constrain `sum(subset) == target`.

**Minimal code.**

```python
import satx

universe = [3, 5, 7, 9]
target = 12

satx.engine(bits=target.bit_length(), cnf_path="tmp_b4.cnf")
bits, subset = satx.subsets(universe)
assert sum(subset) == target

assert satx.satisfy(solver='slime')

subset_values = [v.value for v in subset]
assert sum(subset_values) == target
assert all(subset_values[i] in (0, universe[i]) for i in range(len(universe)))
```

**Scaling notes / pitfalls.**
- `subsets` introduces an internal selection bit-vector and per-element muxes; CNF size grows with `len(universe)`.
- Do not assume a particular “bit means selected” convention; use the masked `subset` values for interpretation.

**Variants.**
- Use a bit tensor and `t[[i]](0, universe[i])` for explicit selection (see Section A1/A4 patterns).

### B5. Balanced partition (`satx.subsets(..., complement=True)`)

**Problem statement.** Partition a multiset into two parts of equal sum.

**Modeling pattern (SATX).**
- Use `bits, sub, com = satx.subsets(data, complement=True)`.
- Constrain `sum(sub) == sum(com)`.

**Minimal code.**

```python
import satx

data = [3, 5, 7, 9]  # total = 24

satx.engine(bits=sum(data).bit_length(), cnf_path="tmp_b5.cnf")
bits, sub, com = satx.subsets(data, complement=True)
assert sum(sub) == sum(com)

assert satx.satisfy(solver='slime')

sv = [v.value for v in sub]
cv = [v.value for v in com]
assert sum(sv) == sum(cv)
assert sum(sv) + sum(cv) == sum(data)
```

**Scaling notes / pitfalls.**
- Complement masks double the number of muxed outputs.
- Add symmetry-breaking constraints if you want a canonical partition (e.g., require the first element to be in `sub`).

**Variants.**
- Enforce a fixed cardinality partition using `satx.subset(data, k=..., complement=True)` (below).

### B6. At most k (non-empty) elements subsets (`satx.subset`)

**Problem statement.** Pick at most `k` items and constrain a linear sum.

**Modeling pattern (SATX).**
- `sub = satx.subset(data, k)` returns a masked list where 1..`k` entries are non-zero (non-empty selection).

**Minimal code.**

```python
import satx

data = [3, 5, 7, 9]

satx.engine(bits=6, cnf_path="tmp_b6.cnf")
sub = satx.subset(data, k=2)
assert sum(sub) == 12

assert satx.satisfy(solver='slime')

sv = [v.value for v in sub]
assert sum(sv) == 12
assert sum(1 for v in sv if v != 0) <= 2
```

**Scaling notes / pitfalls.**
- Internally this uses a combinational cardinality encoding; large `len(data)` and large `k` can blow up CNF.
- Empty selection is not allowed by the current encoding; use `satx.subsets` or explicit bits if you need it.

**Variants.**
- Request the complement too: `sub, com = satx.subset(data, k=..., complement=True)`.

### B7. Selecting with repetition (`satx.combinations`)

**Problem statement.** Choose `n` elements from a small list where repeats are allowed, and constrain their sum.

**Modeling pattern (SATX).**
- `xs, ys = satx.combinations(lst, n)` ties each `ys[i]` to an element of `lst` (repeats allowed).
- Constrain `sum(ys) == target`.

**Minimal code.**

```python
import satx

items = [1, 2, 3]
target = 4

satx.engine(bits=4, cnf_path="tmp_b7.cnf")
xs, ys = satx.combinations(items, n=2)
assert ys[0] + ys[1] == target

assert satx.satisfy(solver='slime')

vals = [v.value for v in ys]
assert sum(vals) == target
assert all(v in items for v in vals)
```

**Scaling notes / pitfalls.**
- `combinations` does not enforce distinctness; use `satx.permutations` if you need a true permutation.

**Variants.**
- Use `satx.permutations(items, n=len(items))` and add linear constraints over `ys`.

## C) Integer feasibility (Diophantine – non-linear)

### C1. Pythagorean triples / circle (`x**2 + y**2 == r**2`)

**Problem statement.** Find integer lattice points on a circle of radius 10.

**Modeling pattern (SATX).**
- Bound variables so intermediate products fit the engine bit-width.
- Use `**2` for squaring and linear sums for the circle equation.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_c1.cnf")

x = satx.integer()
y = satx.integer()

assert x >= 0
assert x <= 10
assert y >= 0
assert y <= 10
assert x**2 + y**2 == 100

assert satx.satisfy(solver='slime')
assert x.value**2 + y.value**2 == 100
```

**Scaling notes / pitfalls.**
- In SATX, arithmetic is over fixed-width bit-vectors; choose `bits` so `x**2`, `y**2`, and their sum fit.
- Multiplication is overflow-checked; overflow makes the model UNSAT.

**Variants.**
- Allow negative coordinates: use `satx.engine(..., signed=True)` and add bounds like `-10 <= x <= 10`.

### C2. Integer factorization (`p*q == n`)

**Problem statement.** Factor a small composite `n` into `p*q`.

**Modeling pattern (SATX).**
- Use multiplication equality as the core constraint.
- Add symmetry breaking (`p <= q`) and parity constraints (oddness via LSB) to reduce search.

**Minimal code.**

```python
import satx

rsa = 77  # 7 * 11

satx.engine(bits=8, cnf_path="tmp_c2.cnf")

p = satx.integer()
q = satx.integer()

assert p > 1
assert q > 1
assert p <= q
assert p * q == rsa

# oddness (LSB == 1)
assert p[[0]](0, 1) == 1
assert q[[0]](0, 1) == 1

assert satx.satisfy(solver='slime')
assert p.value * q.value == rsa
assert (p.value & 1) == 1 and (q.value & 1) == 1
```

**Scaling notes / pitfalls.**
- Factorization is hard; keep `rsa` small unless you are prepared for exponential growth.
- If you omit bounds, SATX still has a finite domain, but the solver may land on trivial factors (e.g., `1*n`) unless excluded.

**Variants.**
- Constrain bit patterns directly (e.g., fix high bits) using `p[[i]](0, 1)` and `q[[i]](0, 1)`.

### C3. Difference of squares (`p**2 - q**2 == n`)

**Problem statement.** Find `p, q` such that `p^2 - q^2 = n` with `q < p`.

**Modeling pattern (SATX).**
- Use squaring plus subtraction; add `q < p` to avoid underflow/wrap in unsigned mode.
- Keep bounds tight so squares fit `bits`.

**Minimal code.**

```python
import satx

n = 21  # 11^2 - 10^2

satx.engine(bits=8, cnf_path="tmp_c3.cnf")

p = satx.integer()
q = satx.integer()

assert p >= 0
assert p <= 15
assert q >= 0
assert q <= 15
assert q < p
assert p**2 - q**2 == n

assert satx.satisfy(solver='slime')
assert p.value**2 - q.value**2 == n
assert q.value < p.value
```

**Scaling notes / pitfalls.**
- In unsigned mode, subtraction does not guard underflow; ensure `q**2 <= p**2` via bounds like `q < p`.
- Squaring is multiplication; overflow is forbidden (UNSAT if it would occur).

**Variants.**
- Reparameterize: `a = p - q`, `b = p + q`, then `a*b == n` (often tighter for factor-like searches).

### C4. Signed products and sign constraints (negative integers)

**Problem statement.** Solve a signed product with known sign: negative times positive is negative.

**Modeling pattern (SATX).**
- Enable two's-complement decoding via `engine(..., signed=True)`.
- Bound ranges to prevent overflow and to force the intended sign region.

**Minimal code.**

```python
import satx

satx.engine(bits=6, cnf_path="tmp_c4.cnf", signed=True)

a = satx.integer()
b = satx.integer()
c = satx.integer()

assert a >= -5
assert a <= -1
assert b >= 1
assert b <= 5
assert c == a * b
assert c < 0

assert satx.satisfy(solver='slime')
assert c.value == a.value * b.value
assert c.value < 0
```

**Scaling notes / pitfalls.**
- Signed addition/subtraction include overflow checks; signed multiplication also constrains the high product bits to sign-extend.
- Division/modulo are unsigned encodings; if you use them in `signed=True`, constrain inputs to be non-negative.

**Variants.**
- Mixed-sign identity: `assert (-a) * b == -(a * b)` (with bounds preventing overflow).

### C5. Absolute difference with negatives (`abs(x - y) == 1`)

**Problem statement.** Find two negative integers with unit distance.

**Modeling pattern (SATX).**
- Use signed engine.
- Use `abs(...)` (signed absolute value is mux-based and constrained non-negative).
- Optionally exclude sentinel-like extremes such as `satx.oo()`.

**Minimal code.**

```python
import satx

satx.engine(bits=6, cnf_path="tmp_c5.cnf", signed=True)

x = satx.integer()
y = satx.integer()

assert x < 0
assert y < 0
assert abs(x - y) == 1
assert x != satx.oo()
assert y != satx.oo()

assert satx.satisfy(solver='slime')
assert abs(x.value - y.value) == 1
assert x.value < 0 and y.value < 0
assert x.value != satx.oo() and y.value != satx.oo()
```

**Scaling notes / pitfalls.**
- In signed mode, `abs(min_int)` is not representable and becomes UNSAT (by overflow checks + non-negativity).
- In unsigned mode, `abs(x)` is not a true absolute value; use `signed=True` for meaningful negatives.

**Variants.**
- Replace with linear form: `assert x - y == 1 or x - y == -1` using `satx.one_of([1, -1])` on the RHS.

### C6. Perfect squares and roots (`satx.sqrt`)

**Problem statement.** Extract an integer square root for a perfect square.

**Modeling pattern (SATX).**
- `satx.sqrt(x)` creates a fresh `y` with the constraint `x == y**2`.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_c6.cnf")

x = satx.constant(81)
y = satx.sqrt(x)

assert satx.satisfy(solver='slime')
assert y.value * y.value == 81
```

**Scaling notes / pitfalls.**
- `satx.sqrt` does not impose a sign convention; in `signed=True`, both `y` and `-y` can satisfy `y**2 == x`.
- If `x` is not a perfect square within the bit-width, the constraint is UNSAT.

**Variants.**
- Force the principal (non-negative) root: `assert y >= 0` (requires `signed=True` if `y` could otherwise go negative).

## D) Exponential / power constraints

### D1. Constant base, variable exponent (`2**n - 7 == x**2`)

**Problem statement.** Solve an exponential Diophantine equation with a variable exponent.

**Modeling pattern (SATX).**
- Use `engine(..., deep=...)` so `base ** exp` is supported when `exp` is a SATX integer.
- Constrain the exponent into the supported range `1..deep` (inclusive).

**Minimal code.**

```python
import satx

satx.engine(bits=8, deep=6, cnf_path="tmp_d1.cnf")

_2 = satx.constant(2)
n = satx.integer()
x = satx.integer()

assert n >= 3
assert n <= 5
assert x >= 0
assert x <= 10
assert _2**n - 7 == x**2

assert satx.satisfy(solver='slime')
assert (2**n.value) - 7 == x.value**2
```

**Scaling notes / pitfalls.**
- Variable exponent encoding selects among `base**i` for `i in 1..deep`; increasing `deep` grows CNF quickly.
- Overflow is forbidden; keep bounds tight on bases and exponents.

**Variants.**
- Use modular exponent: `pow(_2, n, m)` (see Section E).

### D2. Mixed polynomial + variable exponent (`n**3 + 10 == 2**k + 5*n`)

**Problem statement.** Solve a cubic identity with a variable exponent term.

**Modeling pattern (SATX).**
- Use constant exponents (`**3`) directly.
- Use `deep` only for the exponent variable `k` in `2**k`.

**Minimal code.**

```python
import satx

satx.engine(bits=8, deep=6, cnf_path="tmp_d2.cnf")

_2 = satx.constant(2)
n = satx.integer()
k = satx.integer()

assert n >= 0
assert n <= 4
assert k >= 1
assert k <= 6
assert n**3 + 10 == _2**k + 5 * n

assert satx.satisfy(solver='slime')
assert n.value**3 + 10 == (2**k.value) + 5 * n.value
```

**Scaling notes / pitfalls.**
- `n**3` is repeated multiplication; keep `n` small to avoid overflow (which makes the model UNSAT).
- Large `deep` values can dominate solve time even when other parts are small.

**Variants.**
- Replace the cubic with `n**2` when you only need quadratic growth.

### D3. Small power equation (`a**2 == b**3 + 1`)

**Problem statement.** Solve a bounded power equation using only constant exponents.

**Modeling pattern (SATX).**
- Constant exponents use repeated multiplication (no `deep` needed).
- Tight bounds reduce CNF and avoid overflow.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_d3.cnf")

a = satx.integer()
b = satx.integer()

assert a >= 0
assert a <= 10
assert b >= 0
assert b <= 3
assert a**2 == b**3 + 1

assert satx.satisfy(solver='slime')
assert a.value**2 == b.value**3 + 1
```

**Scaling notes / pitfalls.**
- Even with constant exponents, intermediate values must fit within `bits`.

**Variants.**
- Add inequality guards such as `assert a > 0` to exclude trivial solutions.

### D4. Factorials (`satx.factorial`)

**Problem statement.** Recover `n` from `n!` for small `n`.

**Modeling pattern (SATX).**
- `satx.factorial(n)` encodes a product over `n` using an internal selector; it is practical only for small ranges.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_d4.cnf")

n = satx.integer()
f = satx.factorial(n)

assert n >= 1
assert n <= 7
assert f == 120

assert satx.satisfy(solver='slime')
assert n.value == 5
assert f.value == 120
```

**Scaling notes / pitfalls.**
- `factorial` is encoded over the engine `bits`; values outside `0..bits-1` are not representable by its selector scheme.
- Factorials overflow quickly; use small `bits` and small `n`.

**Variants.**
- Use `satx.pi(lambda t: t, i=1, n=n)` to encode product series (below).

### D5. Series sums/products (`satx.sigma`, `satx.pi`)

**Problem statement.** Solve for `n` from series identities like `sum_{t=1..n} t` and `prod_{t=1..n} t`.

**Modeling pattern (SATX).**
- `satx.sigma(f, i, n)` encodes `sum_{t=i..n} f(t)` with `i` a Python `int` and `n` a SATX integer.
- `satx.pi(f, i, n)` similarly encodes a product series.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_d5.cnf")

n = satx.integer()
assert n >= 1
assert n <= 5

s = satx.sigma(lambda t: t, i=1, n=n)  # 1 + 2 + ... + n
p = satx.pi(lambda t: t, i=1, n=n)     # 1 * 2 * ... * n

assert s == 10
assert p == 24

assert satx.satisfy(solver='slime')
assert n.value == 4
assert s.value == 10
assert p.value == 24
```

**Scaling notes / pitfalls.**
- These encodings are selector-based over the engine `bits`; keep `n` small (typically `n <= bits - 2`).
- Products overflow quickly and will make the model UNSAT if the result cannot fit in `bits`.

**Variants.**
- Use `f(t)` that returns SATX expressions (e.g., `lambda t: x + t`) to encode parameterized series.

## E) Modular / congruence-like constraints

### E1. Parity constraints via LSB (`x[[0]](0, 1)`)

**Problem statement.** Constrain a variable to be odd/even.

**Modeling pattern (SATX).**
- Use `x[[0]](0, 1)` to read the least significant bit (0 for even, 1 for odd).

**Minimal code.**

```python
import satx

satx.engine(bits=6, cnf_path="tmp_e1.cnf")

x = satx.integer()
assert x >= 0
assert x <= 15
assert x[[0]](0, 1) == 1  # odd

assert satx.satisfy(solver='slime')
assert (x.value % 2) == 1
```

**Scaling notes / pitfalls.**
- Bit-level constraints are cheap compared to arithmetic `%` encodings.

**Variants.**
- Use `% 2`: `assert x % 2 == 1` (more expensive than a direct bit constraint).
- Use the inverted indicator `satx.switch(x, 0)` (returns 0 when LSB is 1, 1 when LSB is 0).

### E2. Congruences (`x % m == r`)

**Problem statement.** Constrain an integer to a specific residue class.

**Modeling pattern (SATX).**
- Use `%` for remainder constraints (unsigned encoding).
- Bound `x` to keep the search finite and avoid meaningless signed residues.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_e2.cnf")

x = satx.integer()
assert x >= 0
assert x <= 60
assert x % 7 == 3

assert satx.satisfy(solver='slime')
assert x.value % 7 == 3
```

**Scaling notes / pitfalls.**
- `%` is implemented via an unsigned remainder circuit; it is much heavier than bit tests.
- If `signed=True`, treat `%` as unsigned unless you explicitly constrain values to be non-negative.

**Variants.**
- Divisibility: `assert x % d == 0`.

### E3. Modular exponent (bounded) and Fermat-style prime hints (`satx.is_prime`)

**Problem statement.** Search for a small prime candidate using a Fermat-style constraint.

**Modeling pattern (SATX).**
- `satx.is_prime(p)` asserts `pow(2, p, p) == 2` (a base-2 Fermat condition; not a full primality proof).
- Variable exponents require `engine(..., deep=...)`; keep `p` within a small range.

**Minimal code.**

```python
import satx

def is_prime_py(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True

satx.engine(bits=6, deep=8, cnf_path="tmp_e3.cnf")

p = satx.integer()
assert p >= 3
assert p <= 7
assert p[[0]](0, 1) == 1  # odd

satx.is_prime(p)

assert satx.satisfy(solver='slime')
assert is_prime_py(p.value)
```

**Scaling notes / pitfalls.**
- This is a Fermat base-2 condition; composites can pass it (pseudoprimes) outside small ranges.
- Large `deep` values make `pow(base, exp, mod)` expensive because exponentiation uses selection among powers.

**Variants.**
- Force a composite witness: `satx.is_not_prime(p)` (still Fermat-based and not complete).

### E4. Bit rotation constraints (`satx.rotate`)

**Problem statement.** Constrain a variable to be a bit-rotation of another.

**Modeling pattern (SATX).**
- `satx.rotate(x, k)` creates a fresh `v` whose bits are `x` rotated right by `k` (per the implementation).

**Minimal code.**

```python
import satx

satx.engine(bits=4, cnf_path="tmp_e4.cnf")

x = satx.constant(1)        # 0b0001
v = satx.rotate(x, k=1)     # rotate-right by 1 => 0b1000 == 8

assert v == 8
assert satx.satisfy(solver='slime')
assert v.value == 8
```

**Scaling notes / pitfalls.**
- Rotation is purely bit-level; in `signed=True`, it can change sign in non-intuitive ways.

**Variants.**
- Use `tensor(...)` and direct bit constraints when you need multi-word rotations.

### E5. Exact divisibility via `satx.exact_div(...)` / `%`

**Problem statement.** Encode a divisibility constraint and recover a quotient.

**Modeling pattern (SATX).**
- Use `satx.exact_div(x, y)` when you need a quotient and exact divisibility.
- Or constrain exactness explicitly with `assert x % y == 0`.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_e5.cnf")

x = satx.integer()
q = satx.exact_div(x, 3)

assert q == 4
assert satx.satisfy(solver='slime')
assert x.value == 12
assert q.value == 4
```

**Scaling notes / pitfalls.**
- Division is significantly more expensive than addition/multiplication.
- Signed `//`/`%` are built via `abs(...)` + adjustment; if possible, constrain operands non-negative for simpler models.

**Variants.**
- Combine with modulo: `assert x % 3 == 0` and `assert x / 3 == q` (useful when you also need the remainder elsewhere).

## F) Rational constraints (from rational.py)

### F1. Fixed-denominator rational equality (`satx.rational`)

**Problem statement.** Solve `x/8 == 3/4` for integer `x`.

**Modeling pattern (SATX).**
- Build rationals explicitly: `satx.rational(num, den)`.
- Equality is encoded via cross-multiplication with an internal scaling factor; fixing a denominator often forces a unique scaling.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_f1.cnf")

x = satx.integer()
r = satx.rational(x, satx.constant(8))

assert r == satx.rational(satx.constant(3), satx.constant(4))

assert satx.satisfy(solver='slime')
assert x.value == 6
assert x.value / 8 == 3 / 4
```

**Scaling notes / pitfalls.**
- Rational representations are not normalized: `1/2 == 2/4` is allowed by design (via a free scaling factor).
- Keep numerators/denominators bounded and prefer fixed denominators when you want canonical solutions.

**Variants.**
- Mix with integers: `assert satx.rational(x, d) == k` forces `x == k*d` (up to scaling).

### F2. Mixed integer/rational linear equation (`r + 1/2 == 2`)

**Problem statement.** Solve a small linear equation with a rational term.

**Modeling pattern (SATX).**
- Use rational arithmetic (`+`, `-`, `*`) to build a rational expression.
- Compare to an integer to constrain numerator/denominator via the equality encoding.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_f2.cnf")

x = satx.integer()
r = satx.rational(x, satx.constant(2))

assert r + satx.rational(satx.constant(1), satx.constant(2)) == satx.constant(2)

assert satx.satisfy(solver='slime')
assert x.value == 3
assert (x.value / 2) + (1 / 2) == 2
```

**Scaling notes / pitfalls.**
- Avoid calling `repr(...)`, `float(...)`, `abs(...)`, or `invert()` on `Rational` objects before solving; those methods use Python control flow that assumes `.value` is available.

**Variants.**
- Constrain a rational to be an integer: `assert satx.rational(x, d) == satx.integer()`.

### F3. Rational inequalities (cross multiplication)

**Problem statement.** Find `x` such that `1/2 < x/5 < 3/4`.

**Modeling pattern (SATX).**
- Use `>` and `<` on `Rational` to encode cross-multiplied inequalities.
- Keep denominators positive to avoid sign ambiguities.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_f3.cnf", signed=True)

x = satx.integer()
r = satx.rational(x, satx.constant(5))

assert x >= -10
assert x <= 10
assert r > satx.rational(satx.constant(1), satx.constant(2))
assert r < satx.rational(satx.constant(3), satx.constant(4))

assert satx.satisfy(solver='slime')
assert 0.5 < (x.value / 5) < 0.75
```

**Scaling notes / pitfalls.**
- Comparisons are encoded by multiplying across denominators; this can overflow if bounds are loose.
- Keep denominators small and positive when possible.

**Variants.**
- Encode linear rational systems by introducing shared denominators (e.g., all fractions over 12).

### F4. Negative rationals (signed numerators)

**Problem statement.** Solve `n/4 == -3/2` for signed `n`.

**Modeling pattern (SATX).**
- Use `signed=True` and constrain using rational equality.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_f4.cnf", signed=True)

n = satx.integer()
r = satx.rational(n, satx.constant(4))

assert r == satx.rational(satx.constant(-3), satx.constant(2))

assert satx.satisfy(solver='slime')
assert n.value == -6
assert n.value / 4 == -3 / 2
```

**Scaling notes / pitfalls.**
- Signed arithmetic is overflow-checked; keep bounds tight when mixing rationals and products.

**Variants.**
- Constrain denominators to be positive to avoid equivalent sign flips: `(n/d) == (-n)/(-d)`.

## G) Linear algebra / Gaussian helpers (from gaussian.py)

### G1. Gaussian integer multiplication (`(a+bi)*(c+di) == target`)

**Problem statement.** Solve a Gaussian integer factorization in `Z[i]`.

**Modeling pattern (SATX).**
- Use `satx.gaussian()` to create a Gaussian integer with real/imag parts as SATX integers.
- Use Gaussian arithmetic operators; equality decomposes into real and imag constraints.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_g1.cnf", signed=True)

z1 = satx.gaussian()
z2 = satx.gaussian()

assert z1.real >= -5
assert z1.real <= 5
assert z1.imag >= -5
assert z1.imag <= 5
assert z2.real >= -5
assert z2.real <= 5
assert z2.imag >= -5
assert z2.imag <= 5

assert z1 * z2 == satx.gaussian(5, 0)

assert satx.satisfy(solver='slime')

v1 = complex(z1.real.value, z1.imag.value)
v2 = complex(z2.real.value, z2.imag.value)
assert v1 * v2 == 5 + 0j
```

**Scaling notes / pitfalls.**
- This is still fixed-width integer arithmetic underneath; overflow is forbidden.

**Variants.**
- Add a norm bound using `z.real**2 + z.imag**2 <= k`.

### G2. Norm constraints via conjugation (`z * z.conjugate()`)

**Problem statement.** Constrain the squared norm `a^2 + b^2` of a Gaussian integer.

**Modeling pattern (SATX).**
- Use `z.conjugate()` and the identity `z * conj(z) = (a^2 + b^2) + 0j`.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_g2.cnf", signed=True)

z = satx.gaussian()
assert z.real >= -10
assert z.real <= 10
assert z.imag >= -10
assert z.imag <= 10

assert z * z.conjugate() == satx.gaussian(25, 0)

assert satx.satisfy(solver='slime')
assert z.real.value**2 + z.imag.value**2 == 25
```

**Scaling notes / pitfalls.**
- Squaring uses multiplication; bounds are required to prevent overflow.

**Variants.**
- Enumerate all representations of 25 as a sum of two squares by repeated `satx.satisfy(...)`.

### G3. `abs(z)` returns a Gaussian (root selection caveat)

**Problem statement.** Use the provided `abs(z)` for Gaussian integers and constrain the intended root.

**Modeling pattern (SATX).**
- In this repo, `Gaussian.__abs__` returns `Gaussian(sqrt(a^2+b^2), 0)` (a Gaussian number, not a scalar).
- `sqrt(...)` encodes `y**2 == x` and does not force `y >= 0` by itself.

**Minimal code.**

```python
import satx

satx.engine(bits=8, cnf_path="tmp_g3.cnf", signed=True)

z = satx.gaussian()
w = abs(z)

assert w == satx.gaussian(5, 0)  # forces the +5 root

assert satx.satisfy(solver='slime')
assert z.real.value**2 + z.imag.value**2 == 25
```

**Scaling notes / pitfalls.**
- This module is Gaussian integer arithmetic, not Gaussian elimination; for linear systems use dot-products and linear constraints (Section B).

**Variants.**
- If you only need the norm squared, prefer `z * z.conjugate()` to avoid root ambiguity.

## H) gcc.py domain recipes

### H1. Absolute value catalog constraint (`satx.gcc.abs_val`)

**Problem statement.** Relate a signed integer to its absolute value.

**Modeling pattern (SATX).**
- `satx.gcc.abs_val(x, y)` enforces `y >= 0` and `abs(x) == y` (as implemented).

**Minimal code.**

```python
import satx
import satx.gcc as gcc

satx.engine(bits=6, cnf_path="tmp_h1.cnf", signed=True)

x = satx.integer()
y = satx.integer()

assert x >= -7
assert x <= 7
assert x < 0
gcc.abs_val(x, y)

assert satx.satisfy(solver='slime')
assert y.value == abs(x.value)
assert y.value >= 0
```

**Scaling notes / pitfalls.**
- In signed mode, `abs(min_int)` is not representable and will make the model UNSAT.

**Variants.**
- Use `abs(x)` directly if you don't need the `gcc` wrapper.

### H2. Permutation modeling with `gcc.all_different` + `satx.all_in`

**Problem statement.** Constrain a vector to be a permutation of a fixed set.

**Modeling pattern (SATX).**
- Use `satx.all_in(xs, values=[...])` to restrict domains.
- Use `gcc.all_different(xs)` to enforce distinctness.

**Minimal code.**

```python
import satx
import satx.gcc as gcc

satx.engine(bits=3, cnf_path="tmp_h2.cnf")

xs = satx.vector(size=4)
satx.all_in(xs, values=[0, 1, 2, 3])
gcc.all_different(xs)

assert satx.satisfy(solver='slime')
vals = satx.values(xs)
assert sorted(vals) == [0, 1, 2, 3]
```

**Scaling notes / pitfalls.**
- `all_different` is quadratic in the number of variables (pairwise disequalities).

**Variants.**
- Use `gcc.all_equal(xs)` when you need all variables equal instead of distinct.

### H3. Sorting a list into a new list (`satx.gcc.sort`)

**Problem statement.** Constrain `ys` to be the sorted version of `xs`.

**Modeling pattern (SATX).**
- `gcc.sort(xs, ys)` enforces `ys` is a permutation of `xs` and is non-decreasing.

**Minimal code.**

```python
import satx
import satx.gcc as gcc

satx.engine(bits=3, cnf_path="tmp_h3.cnf")

xs = satx.vector(size=4)
ys = satx.vector(size=4)

assert xs[0] == 3
assert xs[1] == 1
assert xs[2] == 2
assert xs[3] == 0

gcc.sort(xs, ys)

assert satx.satisfy(solver='slime')
assert satx.values(ys) == [0, 1, 2, 3]
```

**Scaling notes / pitfalls.**
- Sorting uses permutation encodings; it scales poorly with length.

**Variants.**
- If you also need the permutation indices, use `gcc.sort_permutation` (next recipe).

### H4. Sorting with permutation indices (`satx.gcc.sort_permutation`)

**Problem statement.** Sort values and recover where each sorted element came from.

**Modeling pattern (SATX).**
- `gcc.sort_permutation(lst_from, lst_per, lst_to)` constrains `lst_to` sorted and `lst_per[i]` to be the source index for `lst_to[i]`.

**Minimal code.**

```python
import satx
import satx.gcc as gcc

satx.engine(bits=3, cnf_path="tmp_h4.cnf")

src = satx.vector(size=4)
per = satx.vector(size=4)
dst = satx.vector(size=4)

assert src[0] == 3
assert src[1] == 1
assert src[2] == 2
assert src[3] == 0

gcc.sort_permutation(src, per, dst)

assert satx.satisfy(solver='slime')

src_v = satx.values(src)
per_v = satx.values(per)
dst_v = satx.values(dst)

assert dst_v == sorted(src_v)
assert all(src_v[per_v[i]] == dst_v[i] for i in range(len(dst_v)))
```

**Scaling notes / pitfalls.**
- `lst_per` is itself a SATX vector of integers; keep it bounded if you add extra arithmetic constraints.

**Variants.**
- Use `satx.permutations(...)` directly if you want the raw permutation variables without a sorting constraint.

### H5. Hamming distance between vectors (`satx.gcc.all_differ_from_exactly_k_pos`)

**Problem statement.** Constrain two binary vectors to differ in exactly `k` positions.

**Modeling pattern (SATX).**
- Build a list of vectors (same length).
- Use `gcc.all_differ_from_exactly_k_pos(k, vectors)` to constrain pairwise Hamming distance (as implemented).

**Minimal code.**

```python
import satx
import satx.gcc as gcc

satx.engine(bits=3, cnf_path="tmp_h5.cnf")

v0 = satx.vector(size=4)
v1 = satx.vector(size=4)
satx.all_binaries(v0)
satx.all_binaries(v1)

gcc.all_differ_from_exactly_k_pos(2, [v0, v1])

assert satx.satisfy(solver='slime')

a = satx.values(v0)
b = satx.values(v1)
assert sum(x != y for x, y in zip(a, b)) == 2
```

**Scaling notes / pitfalls.**
- These constraints introduce auxiliary tensors and scale with (#vectors^2 * length).

**Variants.**
- Use `all_differ_from_at_least_k_pos` or `all_differ_from_at_most_k_pos` to relax the distance condition.

### H6. Counting occurrences (lower-bound semantics) (`satx.gcc.count`)

**Problem statement.** Ensure at least `k` elements in a binary vector equal a given value.

**Modeling pattern (SATX).**
- `gcc.count(val, lst, rel, lim)` uses internal selector bits; with `rel` as `>=`, the constraint acts like "at least lim occurrences".

**Minimal code.**

```python
import satx
import satx.gcc as gcc

satx.engine(bits=3, cnf_path="tmp_h6.cnf")

xs = satx.vector(size=4)
satx.all_binaries(xs)

gcc.count(val=0, lst=xs, rel=lambda a, b: a >= b, lim=2)

assert satx.satisfy(solver='slime')
vals = satx.values(xs)
assert sum(v == 0 for v in vals) >= 2
```

**Scaling notes / pitfalls.**
- As implemented, `count(..., rel=lambda a,b: a == b, lim=k)` enforces "at least k occurrences", not "exactly k".

**Variants.**
- For exact counting, prefer explicit indicator variables (Section A1/A4) and sum them.

## I) Optimization patterns (iterative bounding)

### I1. Enumerating all models (blocking clause behavior)

**Problem statement.** Enumerate all values of a bounded variable.

**Modeling pattern (SATX).**
- Each successful `satx.satisfy(...)` appends a blocking clause forbidding the current model.
- Repeated calls enumerate distinct solutions until UNSAT.

**Minimal code.**

```python
import satx

satx.engine(bits=3, cnf_path="tmp_i1.cnf")

x = satx.integer()
assert x >= 0
assert x <= 3

seen = set()
while satx.satisfy(solver='slime'):
    seen.add(x.value)
    satx.clear([x])

assert seen == {0, 1, 2, 3}
```

**Scaling notes / pitfalls.**
- CNF grows by one blocking clause per model; enumeration is only practical for small spaces.

**Variants.**
- Enumerate a vector and block a tuple by adding a custom clause over the bits you care about.

### I2. Maximize a variable by iterative tightening

**Problem statement.** Maximize `x` subject to constraints, without a built-in optimizer.

**Modeling pattern (SATX).**
- Solve once to get a feasible `x`.
- Add a stronger bound `x > best` and solve again.
- Repeat until UNSAT; last `best` is the maximum.

**Minimal code.**

```python
import satx

satx.engine(bits=4, cnf_path="tmp_i2.cnf")

x = satx.integer()
assert x >= 0
assert x <= 7
assert 2 * x <= 10

best = None
while satx.satisfy(solver='slime'):
    best = x.value
    satx.clear([x])
    assert x > best

assert best == 5
```

**Scaling notes / pitfalls.**
- Tightening adds constraints and blocking clauses into the same CNF; for many iterations, consider rebuilding the model from scratch each round.

**Variants.**
- Minimize instead by adding `assert x < best` after each SAT model.

### I3. Minimize in signed space (two-solution example)

**Problem statement.** Minimize a signed variable when multiple symmetric solutions exist.

**Modeling pattern (SATX).**
- Use `signed=True` and iterative tightening with `<`.

**Minimal code.**

```python
import satx

satx.engine(bits=6, cnf_path="tmp_i3.cnf", signed=True)

x = satx.integer()
assert x**2 == 4  # solutions: x == 2 or x == -2

best = None
while satx.satisfy(solver='slime'):
    best = x.value
    satx.clear([x])
    assert x < best

assert best == -2
```

**Scaling notes / pitfalls.**
- Signed comparisons use two's-complement ordering; keep bounds small to avoid accidental overflow in intermediate expressions.

**Variants.**
- Lexicographic optimization for vectors: optimize `x0`, fix it, then optimize `x1`, etc. (build/rebuild between stages to avoid CNF growth).

## Appendix

### Modeling checklist

- Initialize once per problem: `satx.engine(bits=..., deep=..., cnf_path=..., signed=...)`.
- Declare variables using SATX constructors (`satx.integer`, `satx.vector`, `satx.tensor`, …).
- Bound every variable with `assert` (especially before `*`, `**`, `%`, `//`, `pow`).
- Write constraints as expressions that get *evaluated* (usually via `assert ...`); avoid Python control flow (`if`, `and`, `or`).
- Solve with `assert satx.satisfy(solver='slime')`, then read `.value` (or `satx.values(...)`) for sanity checks.
- For multiple models, call `satx.satisfy(...)` repeatedly (blocking clause is automatic) and clear cached values with `satx.clear([...])`.

### Choosing bit-width / bounds

- Unsigned (`signed=False`): values decode in `[0, 2**bits - 1]`.
- Signed (`signed=True`): values decode in two’s complement `[-2**(bits-1), 2**(bits-1) - 1]`; `satx.oo()` is the max positive in that mode.
- Overflow is forbidden for `+` and `*` (and for signed `-` as well); if an expression cannot fit, the instance becomes UNSAT.
- Unsigned subtraction can underflow (wrap) unless you constrain it away (e.g., ensure `a >= b` before asserting `a - b == ...`).
- Variable exponents require `engine(..., deep=...)` and `exp > 0`; the implementation selects among `base**i` for `i in 1..deep`.
- **Known limitation (as implemented):** with `signed=True`, any `Unit` of bit-width 1 causes decoding failure in `satx.satisfy(...)`; avoid `satx.integer(bits=1)`/1-bit tensors in signed problems.

### Debugging UNSAT

- Add bounds first, then solve early; tighten constraints incrementally.
- If a constraint mixes domains (e.g., `%`, `//`, `pow`), test it in isolation with very small bounds.
- Use `cnf_path=...` to keep the emitted CNF; `satx.satisfy(...)` also writes `<key>.mod` with solver output.
- Use `satx.satisfy(solver='slime', log=True)` to stream solver output while debugging.
- If you’re iterating with repeated `satx.satisfy(...)`, remember the CNF accumulates blocking clauses and new constraints; rebuild with a fresh `satx.engine(...)` when needed.

### Performance traps

- Large `deep` + variable exponentiation (`base ** exp` where `exp` is a SATX integer) dominates CNF size.
- Large tables in `satx.index(...)`, `satx.element(...)`, and large choice sets in `satx.one_of(...)` scale linearly with table size and bit-width.
- Permutations/sorting (`satx.permutations`, `satx.gcc.sort`, `satx.gcc.sort_permutation`) scale poorly with length.
- Non-linear constraints (factorization, products of unknowns) can become hard even for small bounds.
- Repeated-model loops (`while satx.satisfy(...): ...`) grow the CNF by one blocking clause per model; for long loops, rebuild instead of accumulating.

### When to use gaussian/rational helpers

- Use `satx.gaussian()` / `satx.Gaussian` when the problem is naturally complex-valued but still integer (real/imag are ordinary SATX integers).
- Use `satx.rational()` / `satx.Rational` to express fractions via numerator/denominator constraints; prefer fixed denominators to avoid many equivalent representations.
- As implemented, avoid calling `Rational.invert()`, `Rational.__truediv__`, `abs(r)`, `float(r)`, or `repr(r)` before solving; these methods use Python control flow that only behaves correctly after `.value` is assigned.
- `Gaussian.__abs__` returns a Gaussian number `(sqrt(a^2+b^2) + 0j)`, not a scalar; constrain the intended root explicitly (Section G3).

### Public helper map (coverage audit)

This section enumerates all user-facing callables exported by `satx` (52) and `satx.gcc` (11) in this repo, with a short description and a pointer to a runnable example section.

**`satx` (from `satx/stdlib.py` plus exported classes)**
- `satx.ALU` — low-level CNF builder used by `satx.engine` (`satx/alu.py`); most users should not instantiate it directly. Example: [Template](#template-copypaste).
- `satx.Gaussian` — Gaussian integer type with `.real`/`.imag` parts (`satx/gaussian.py`). Example: [G](#g-linear-algebra--gaussian-helpers-from-gaussianpy).
- `satx.Rational` — rational type with `.numerator`/`.denominator` (`satx/rational.py`). Example: [F](#f-rational-constraints-from-rationalpy).
- `satx.Unit` — core fixed-width integer/bit-vector type created by `satx.integer/constant/...` (`satx/unit.py`). Examples: throughout.
- `satx.all_binaries(lst)` — constrain each element of `lst` to `0..1` (`satx/stdlib.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.all_different(args)` — pairwise disequalities for all elements (`satx/stdlib.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.all_in(args, values)` — constrain each element to be in `values` (`satx/stdlib.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.all_out(args, values)` — constrain each element to be different from each value in `values` (`satx/stdlib.py`). No dedicated recipe; see [Modeling checklist](#modeling-checklist).
- `satx.apply_different(lst, f, indexed=False)` — apply `f` to all ordered pairs `i != j` (`satx/stdlib.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.apply_dual(lst, f, indexed=False)` — apply `f` to all pairs `i < j` (`satx/stdlib.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.apply_single(lst, f, indexed=False)` — apply `f` to each element (`satx/stdlib.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.at_most_k(x, k)` — constrain a selector bit-vector so at most `k` bits are false (selected) and at least one is false (`satx/stdlib.py`). Example: [B](#b-integer-feasibility-diophantine--linear).
- `satx.bits()` — return engine bit-width (`satx/stdlib.py`). Example: [E](#e-modular--congruence-like-constraints).
- `satx.check_engine()` — abort if the engine was not initialized (`satx/stdlib.py`). Example: [Modeling checklist](#modeling-checklist).
- `satx.clear(lst)` — call `.clear()` on each unit (clears cached `.value`) (`satx/stdlib.py`). Example: [I](#i-optimization-patterns-iterative-bounding).
- `satx.combinations(lst, n)` — entangle `n` selections from `lst` (repeats allowed): returns `(xs, ys)` (`satx/stdlib.py`). Example: [B](#b-integer-feasibility-diophantine--linear).
- `satx.constant(value, bits=None)` — create a constant `Unit` (`satx/stdlib.py`). Example: [C](#c-integer-feasibility-diophantine--non-linear).
- `satx.dot(xs, ys)` — dot product helper (`satx/stdlib.py`). Example: [B](#b-integer-feasibility-diophantine--linear).
- `satx.element(item, data)` — return index variable for `item` in `data` (`satx/stdlib.py`). Example: [A](#a-boolean--cnf-ish-modeling-patterns-bridge-layer).
- `satx.engine(bits=None, deep=None, info=False, cnf_path='', signed=False, simplify=False)` — initialize/reset global engine (`satx/stdlib.py`). Example: [Template](#template-copypaste).
- `satx.factorial(x)` — factorial encoding for small `x` (`satx/stdlib.py`). Example: [D](#d-exponential--power-constraints).
- `satx.flatten(mtx)` — pure-Python flatten helper (no SAT constraints) (`satx/stdlib.py`). No dedicated recipe; see [Modeling checklist](#modeling-checklist).
- `satx.gaussian(x=None, y=None)` — Gaussian constructor helper (`satx/stdlib.py`). Example: [G](#g-linear-algebra--gaussian-helpers-from-gaussianpy).
- `satx.hess_abstract(xs, oracle, f, g, log=None, fast=False, cycles=1, target=0)` — pure-Python heuristic search (`satx/stdlib.py`). No SAT example; see [Performance traps](#performance-traps).
- `satx.hess_binary(n, oracle, fast=False, cycles=1, target=0, seq=None)` — pure-Python heuristic search over bit-vectors (`satx/stdlib.py`). No SAT example; see [Performance traps](#performance-traps).
- `satx.hess_sequence(n, oracle, fast=False, cycles=1, target=0, seq=None)` — pure-Python heuristic search over sequences (`satx/stdlib.py`). No SAT example; see [Performance traps](#performance-traps).
- `satx.hyper_loop(n, m)` — pure-Python nested-loop generator over `range(m)^n` (`satx/stdlib.py`). No dedicated recipe; see [Modeling checklist](#modeling-checklist).
- `satx.index(ith, data)` — return a variable constrained to `data[ith]` (`satx/stdlib.py`). Example: [A](#a-boolean--cnf-ish-modeling-patterns-bridge-layer).
- `satx.integer(bits=None)` — create a fresh integer variable (`satx/stdlib.py`). Example: [B](#b-integer-feasibility-diophantine--linear).
- `satx.is_not_prime(p)` — Fermat-style non-primality hint (base 2) (`satx/stdlib.py`). Example: [E](#e-modular--congruence-like-constraints).
- `satx.is_prime(p)` — Fermat-style primality hint (base 2) (`satx/stdlib.py`). Example: [E](#e-modular--congruence-like-constraints).
- `satx.matrix(bits=None, dimensions=None, is_gaussian=False, is_rational=False)` — nested-list constructor; with `is_gaussian/is_rational`, each logical cell appends both the object and the last scalar component (row length doubles), as implemented (`satx/stdlib.py`). No dedicated recipe; see [Modeling checklist](#modeling-checklist).
- `satx.matrix_permutation(lst, n)` — permutation-driven indexing into a flattened `n×n` table (`satx/stdlib.py`). No dedicated recipe; see [Modeling checklist](#modeling-checklist).
- `satx.mul(xs, ys)` — elementwise multiplication helper (`satx/stdlib.py`). No dedicated recipe; see dot patterns in [B](#b-integer-feasibility-diophantine--linear).
- `satx.one_of(lst)` — exact-one choice that returns the chosen value (`satx/stdlib.py`). Example: [A](#a-boolean--cnf-ish-modeling-patterns-bridge-layer).
- `satx.oo()` — max representable value (unsigned max or signed max positive) (`satx/stdlib.py`). Example: [C](#c-integer-feasibility-diophantine--non-linear).
- `satx.permutations(lst, n)` — entangle a permutation of `lst`: returns `(xs, ys)` (`satx/stdlib.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.pi(f, i, n)` — product series encoding (`satx/stdlib.py`). Example: [D](#d-exponential--power-constraints).
- `satx.rational(x=None, y=None)` — rational constructor helper (`satx/stdlib.py`). Example: [F](#f-rational-constraints-from-rationalpy).
- `satx.reset()` — delete the current CNF file and reset render state (`satx/stdlib.py`). Example: [Template](#template-copypaste).
- `satx.reshape(lst, dimensions)` — pure-Python reshape helper (no SAT constraints) (`satx/stdlib.py`). No dedicated recipe; see [Modeling checklist](#modeling-checklist).
- `satx.rotate(x, k)` — bit-rotation constraint (`satx/stdlib.py`). Example: [E](#e-modular--congruence-like-constraints).
- `satx.satisfy(solver, params='', log=False)` — run solver, decode model, add blocking clause (`satx/stdlib.py`). Examples: throughout.
- `satx.sigma(f, i, n)` — sum series encoding (`satx/stdlib.py`). Example: [D](#d-exponential--power-constraints).
- `satx.sqrt(x)` — introduce `y` with `x == y**2` (`satx/stdlib.py`). Example: [C](#c-integer-feasibility-diophantine--non-linear).
- `satx.subset(lst, k, empty=None, complement=False)` — masked subset with at most `k` selected elements (non-empty) (`satx/stdlib.py`). Example: [B](#b-integer-feasibility-diophantine--linear).
- `satx.subsets(lst, k=None, complement=False)` — selection bits + masked subset (optional cardinality) (`satx/stdlib.py`). Example: [B](#b-integer-feasibility-diophantine--linear).
- `satx.switch(x, ith, neg=False)` — inverted bit indicator helper (`satx/stdlib.py`). Example: [A](#a-boolean--cnf-ish-modeling-patterns-bridge-layer).
- `satx.tensor(dimensions)` — create a tensor/bit-array `Unit` (`satx/stdlib.py`). Example: [A](#a-boolean--cnf-ish-modeling-patterns-bridge-layer).
- `satx.values(lst, cleaner=None)` — extract `.value` list (optional filter) (`satx/stdlib.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.vector(bits=None, size=None, is_gaussian=False, is_rational=False)` — construct a vector of ints/gaussians/rationals (`satx/stdlib.py`). Example: [B](#b-integer-feasibility-diophantine--linear).
- `satx.version()` — print system banner (`satx/stdlib.py`). No SAT example; see [Modeling checklist](#modeling-checklist).

**`satx.gcc` (from `satx/gcc.py`)**
- `satx.gcc.abs_val(x, y)` — constrain `y >= 0` and `abs(x) == y` (as implemented). Example: [H](#h-gccpy-domain-recipes).
- `satx.gcc.all_differ_from_at_least_k_pos(k, lst)` — pairwise Hamming distance at least `k` (as implemented). Example: [H](#h-gccpy-domain-recipes).
- `satx.gcc.all_differ_from_at_most_k_pos(k, lst)` — pairwise Hamming distance at most `k` (as implemented). Example: [H](#h-gccpy-domain-recipes).
- `satx.gcc.all_differ_from_exactly_k_pos(k, lst)` — pairwise Hamming distance exactly `k` (as implemented). Example: [H](#h-gccpy-domain-recipes).
- `satx.gcc.all_different(lst)` — alias to `satx.all_different` (`satx/gcc.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.gcc.all_equal(lst)` — constrain all elements equal (`satx/gcc.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.gcc.count(val, lst, rel, lim)` — occurrence lower-bound pattern using internal selectors (`satx/gcc.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.gcc.element(idx, lst, val)` — wrapper: `val == satx.index(idx, lst)` (`satx/gcc.py`). Example: [A](#a-boolean--cnf-ish-modeling-patterns-bridge-layer).
- `satx.gcc.gcd(x, y, z)` — GCD-like constraint pattern (see implementation) (`satx/gcc.py`). Example: [E](#e-modular--congruence-like-constraints).
- `satx.gcc.sort(lst1, lst2)` — sorted permutation constraint (`satx/gcc.py`). Example: [H](#h-gccpy-domain-recipes).
- `satx.gcc.sort_permutation(lst_from, lst_per, lst_to)` — sorted permutation with permutation indices (`satx/gcc.py`). Example: [H](#h-gccpy-domain-recipes).

**Coverage audit.** The lists above match the public callable sets in this repo (`satx`: 52, `satx.gcc`: 11). Every symbol is named here with a pointer to an example section in this cookbook.
