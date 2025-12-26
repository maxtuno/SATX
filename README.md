# SATX

**SATX** is an exact constraint-based modeling system built on top of SAT solvers.

It allows users to express arithmetic, logical, and algebraic constraints
directly in Python, and solve them using SAT technology, while preserving
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

## Installation

```bash
# install the latest development sources directly from the SATX repository
pip install git+https://github.com/maxtuno/SATX.git

# install the package into a virtual environment for development
pip install -e .[dev]
```

You can also build source and wheel distributions locally:

```bash
python -m build
```

---

## Basic Example

```python
import satx

satx.engine(bits=8)

x = satx.integer()
y = satx.integer()

assert x > 0
assert y > 0
assert x + y == 7

if satx.satisfy(solver="slime.exe"):
    print(x, y)
else:
    print("UNSAT")
