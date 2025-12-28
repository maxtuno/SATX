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

---

### Problem Statement

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

### Computational Approach

The variables are encoded as bounded signed integers, and the equation is translated into a Boolean satisfiability (SAT) problem. The solver searches for assignments that satisfy:

- Nonzero bases (a, b, c),
- Positive exponents (x, y, z, e),
- Exact equality of the exponential expression.

The search is exhaustive within the chosen bit-width and reports all satisfying assignments until the formula becomes unsatisfiable.

---

### Example Solution (Bounded Model)

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

### Remarks

* The presence of negative bases introduces parity effects that allow nontrivial solutions even for large exponents.
* In the unrestricted (unbounded) setting, the existence of nontrivial solutions is largely open.
* This formulation serves as a **benchmark problem** for studying the limits of SAT solvers on arithmetic with extreme nonlinear growth.

---


