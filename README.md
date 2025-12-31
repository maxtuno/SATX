# SATX

**SATX** is an exact constraint-based modeling system built on top of SAT solvers.

It allows users to express arithmetic, logical, and algebraic constraints
directly in Python and solve them using SAT technology, while preserving
discrete exactness.

SATX is designed for researchers, engineers, and advanced users who need
precise constraint modeling beyond floating-point optimization.

---

## What SATX Is

SATX provides:

- Integer and rational constraint modeling
- Exact arithmetic (no floating-point semantics)
- Support for negative values
- Linear and non-linear Diophantine constraints
- Algebraic helpers (Gaussian-style, rational constraints)
- SAT-based feasibility solving

SATX is **not** a traditional MIP solver.  
It prioritizes exactness and expressiveness over numerical relaxation.

---

## Installation

```bash
pip install git+https://github.com/maxtuno/SATX.git
```

## Documentation

- Full reference: `docs/satx_full_reference.md`
- Math cookbook: `docs/cookbook_math.md`
- Fixed-point release notes: `docs/fixed_point_release_notes.md`

## Roadmap
See 'docs/ROADMAP.md' for scope, milestones, and what “verification” means in SATX (SAT/UNSAT + certificates).

## Prerequisites

SATX emits CNF and calls an external SAT solver.
- Ensure a solver binary is on PATH (examples use `slime` via `satx.satisfy(solver="slime")`).

---

## Quickstart

```python
import satx

satx.engine(bits=8, signed=True, cnf_path='test.cnf')

x = satx.integer()
y = satx.integer()

assert x > 0
assert y > 0
assert x + y == 7

if satx.satisfy(solver="slime"):
    print(x, y)
else:
    print("UNSAT")
```

If you omit `cnf_path`, SATX auto-generates `<script>.cnf` from the current script name.
(Note: `cnf_path=""` triggers legacy behavior and currently raises "No cnf file specified...".)

--- 

## Fixed-point decimals (scaled integers)

SATX avoids floating-point arithmetic. Use `satx.Fixed` to model decimal values as a scaled integer (`raw / scale`).

```python
import satx
from fractions import Fraction

satx.engine(bits=16, cnf_path="tmp_fixed.cnf")

a = satx.fixed_const(1.25, scale=100)
b = satx.fixed_const(2.00, scale=100)
c = satx.fixed_const(2.50, scale=100)

assert a * b == c
assert satx.satisfy(solver="slime")
assert c.value == Fraction(5, 2)  # 2.50
```

You can also construct fixed-point values directly via `satx.integer(scale=...)`:

```python
import satx

satx.engine(bits=16, cnf_path="tmp_fixed_integer.cnf")
x = satx.integer(scale=100)
y = satx.integer(scale=100)
total = satx.integer(scale=100)
assert x + y == total
assert satx.satisfy(solver="slime")
```

Fixed-by-default mode (opt-in):

```python
import satx

satx.engine(bits=12, fixed_default=True, fixed_scale=100, cnf_path="tmp_fixed_default.cnf")
x = satx.integer()            # Fixed(scale=100)
y = satx.vector(size=3)       # list[Fixed]
u = satx.integer(force_int=True)  # Unit override
z = satx.vector(size=3, fixed=True, scale=1000)
```

Need scale guidance? `satx.fixed_advice(...)` returns numeric limits (no constraints, no output).

Printing:
- `str(fixed)` and `repr(fixed)` are numeric (lists/matrices print clean values).
- For diagnostics, use `fixed.debug_repr()`.

## Examples

### Goormaghtigh equation

#### Statement

The **Goormaghtigh equation** is the exponential Diophantine equation

$$
\frac{x^{m} - 1}{x - 1} \;=\; \frac{y^{n} - 1}{y - 1},
$$

where

- \(x, y > 1\) are integers,
- \(m, n > 2\) are integers,
- and \(x \neq y\).

Equivalently, it can be written as

$$
1 + x + x^2 + \cdots + x^{m-1}
\;=\;
1 + y + y^2 + \cdots + y^{n-1}.
$$

#### Interpretation

The equation states that two **finite geometric series** with different bases have the same sum.  
Each side represents a *repunit-like* number expressed in base \(x\) and base \(y\), respectively.

#### Known results

Only two non-trivial solutions are currently known:

$$
\frac{2^{5} - 1}{2 - 1} = \frac{5^{3} - 1}{5 - 1} = 31,
$$

$$
\frac{2^{13} - 1}{2 - 1} = \frac{90^{3} - 1}{90 - 1} = 8191.
$$

#### Status

It is **conjectured** that these are the only solutions with \(x \neq y\) and \(m,n > 2\).  
Despite its simple appearance, the equation remains **unsolved in general**.

#### Mathematical significance

The Goormaghtigh equation lies at the intersection of:
- exponential Diophantine equations,
- geometric progressions,
- and the theory of linear forms in logarithms.

It serves as a benchmark problem illustrating how elementary-looking equations can encode deep arithmetic complexity.

---

```python
import satx

satx.engine(bits=16, signed=True, simplify=True, cnf_path='test.cnf')

m = satx.integer()
n = satx.integer()

x = satx.integer()
y = satx.integer()

assert (x ** m - 1) // (x - 1) == (y ** n - 1) // (y - 1)

assert m > 2
assert n > 2

 # To make it more interesting for showcasing the potential of SATX.

assert abs(x) > 1
assert abs(y) > 1

assert x < y

while satx.satisfy(solver='slime', params='-use-distance -cryptography', log=False):
    print('(m, n):', m, n)
    print('(x, y):', x, y)
    print('Is Valid?', (x ** m - 1) // (x - 1), ' == ', (y ** m - 1) // (y - 1))
    print()
else:
    print("UNSAT")

"""
Output:
(m, n): 3 3
(x, y): -3 2
Is Valid? 7  ==  7

UNSAT
"""
```

Note

This problem may take a long time to solve. Its difficulty depends on the chosen
bit-size parameter and on the intrinsic complexity of the problem.
Some problems are NP-hard, and in certain cases even undecidable when
considered over infinite domains.

However, advances in SAT solvers have made SAT-based approaches
significantly more powerful. In practice, any standard solver that performs well
in the SAT Competition can be used effectively.

---

### Super-exponential Diophantine equation (polynomial exponents)

#### Problem statement

**Exponential Diophantine Equation with Polynomial Exponents**

We consider the following exponential Diophantine equation:

$$
a^{x^{e}} + b^{y^{e}} = c^{z}
$$

where

$$
a,b,c \in \mathbb{Z} \setminus \{0\}, \qquad x,y,z,e \in \mathbb{Z}_{>0}.
$$

The objective is to determine whether there exist integer solutions satisfying the equation under bounded arithmetic, and to enumerate such solutions if they exist.

This problem belongs to the class of **super-exponential Diophantine equations**, where the growth rate is dominated by polynomial exponents inside exponential terms. Such equations are generally intractable by classical analytic methods and are therefore explored here using **SAT-based bounded model checking**.

---

#### Computational approach

The variables are encoded as bounded signed integers, and the equation is translated into a Boolean satisfiability (SAT) problem. The solver searches for assignments that satisfy:

- Nonzero bases (a, b, c),
- Positive exponents (x, y, z, e),
- Exact equality of the exponential expression.

The search is exhaustive within the chosen bit-width and reports all satisfying assignments until the formula becomes unsatisfiable.

---

#### Example solution (bounded model)

One solution found by the solver is:

$$
a = 3,\; b = -2,\; c = -1,\; x = 1,\; y = 1,\; z = 8,\; e = 6
$$

which satisfies:

$$
3^{1} + (-2)^{1} = (-1)^{8}
$$

that is,

$$
3 - 2 = 1
$$

```python
import satx

satx.engine(bits=16, signed=True, cnf_path='test.cnf')

a = satx.integer()
b = satx.integer()
c = satx.integer()

x = satx.integer()
y = satx.integer()
z = satx.integer()

e = satx.integer()

assert a ** (x ** e) + b ** (y ** e) == c ** z

satx.apply_single([a, b, c], lambda k: k != 0)
satx.apply_single([x, y, z], lambda k: k > 0)

while satx.satisfy(solver='slime'):
    print(a, b, c, x, y, z, e)
else:
    print("UNSAT")
```

##### Note: 
$$ 
2^{2^{3}} + 2^{2^{3}} = 2^{9} 
$$

#### Remarks

* The presence of negative bases introduces parity effects that allow nontrivial solutions even for large exponents.
* In the unrestricted (unbounded) setting, the existence of nontrivial solutions is largely open.
* This formulation serves as a **benchmark problem** for studying the limits of SAT solvers on arithmetic with extreme nonlinear growth.


### Facility location / assignment (small MIP-style example)

Problem statement (plain English):
- We have 3 candidate facilities and 3 clients.
- Opening a facility `i` has a fixed cost `f[i]`.
- Assigning client `j` to facility `i` has a cost `c[j][i]`.
- Each client must be assigned to exactly one facility.
- A client can only be assigned to a facility if that facility is opened.
- At least one facility must be opened.
- Decision variables are binary.

```python
import satx

satx.version()

satx.engine(bits=10, cnf_path="tmp_facility.cnf")

# y[i] = 1 if facility i is opened
y = satx.vector(size=3)

# x[j][i] = 1 if client j is assigned to facility i
x = satx.matrix(dimensions=(3, 3))

cost_open = [9, 7, 6]
cost_assign = [[4, 6, 9], [5, 4, 7], [6, 3, 4]]

obj = satx.dot(cost_open, y) + satx.dot(satx.flatten(x), satx.flatten(cost_assign))

for j in range(3):
    assert sum(x[j][i] for i in range(3)) == 1

for j in range(3):
    for i in range(3):
        assert x[j][i] <= y[i]

assert sum(y) >= 1

satx.all_binaries(y)
satx.all_binaries(satx.flatten(x))

optimal = satx.oo()
while satx.satisfy(solver="slime"):
    print(obj)
    optimal = obj.value
    satx.clear([obj])
    assert obj < optimal
else:
    print("Final best objective value")
    print(x)
    print(y)
    print(optimal)
```

---

## License

See `LICENSE`.
