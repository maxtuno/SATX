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
pip install -e .[dev]
```

```bash
python -m build
```

---

## Basic Example

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

---

# Goormaghtigh Equation

## Statement

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

## Interpretation

The equation states that two **finite geometric series** with different bases have the same sum.  
Each side represents a *repunit-like* number expressed in base \(x\) and base \(y\), respectively.

## Known Results

Only two non-trivial solutions are currently known:

$$
\frac{2^{5} - 1}{2 - 1} = \frac{5^{3} - 1}{5 - 1} = 31,
$$

$$
\frac{2^{13} - 1}{2 - 1} = \frac{90^{3} - 1}{90 - 1} = 8191.
$$

## Status

It is **conjectured** that these are the only solutions with \(x \neq y\) and \(m,n > 2\).  
Despite its simple appearance, the equation remains **unsolved in general**.

## Mathematical Significance

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


