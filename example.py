"""
Facility Staffing / Assignment (Small MIP) — SATX implementation
---------------------------------------------------------------

Problem statement (plain English)
We have 2 candidate service centers and 3 clients.
Opening center i has a fixed cost f_i.
Assigning client j to center i has cost c_{j,i}.
Each client j has a (decimal) workload demand d_j hours.
If center i is opened, we can hire an integer number of technicians s_i (>=0).
Each technician provides H = 8.0 hours of capacity.
Each client must be assigned to exactly one center.
A client can only be assigned to an opened center.
Each center's assigned workload must not exceed its capacity.
At least one center must be opened.

Standard MIP formulation

Sets
  I = {1,2}  (centers)
  J = {1,2,3} (clients)

Parameters (decimals)
  f_i     fixed open cost
  w_i     cost per technician
  c_{j,i} assignment cost
  d_j     demand hours
  H       hours per technician (= 8.0)

Decision variables
  y_i ∈ {0,1}       open center i
  x_{j,i} ∈ {0,1}   assign client j to center i
  s_i ∈ Z_{>=0}     technicians hired at center i

Objective
  minimize  Σ_i f_i y_i + Σ_i w_i s_i + Σ_j Σ_i c_{j,i} x_{j,i}

Constraints
  (1) ∀j: Σ_i x_{j,i} = 1
  (2) ∀j,i: x_{j,i} ≤ y_i
  (3) ∀i: Σ_j d_j x_{j,i} ≤ H s_i
  (4) ∀i: s_i ≤ M y_i   (use M=3)
  (5) Σ_i y_i ≥ 1

NOTE (decimals in SATX):
This script uses fixed-point scaling by 10 (1 decimal digit).
So: 5.0 -> 50, 3.2 -> 32, 6.5 -> 65, H=8.0 -> 80, etc.
The reported objective is also scaled by 10.

Known optimum for this instance:
  y = [0,1]
  s = [0,3]
  assign all clients to center 2
  objective = 17.1
"""

import satx

print(satx.version())

# -----------------------------
# SATX engine configuration
# -----------------------------
satx.engine(bits=10, fixed_default=True, fixed_scale=10)  # enough for small scaled sums

# -----------------------------
# Decision variables
# -----------------------------
# y[i] = 1 if center i is opened (2 centers)
y = satx.vector(size=2)

# x[j][i] = 1 if client j assigned to center i (3x2)
x = satx.matrix(dimensions=(3, 2))

# s[i] = integer technicians at center i (0..3)
s = satx.vector(size=2, fixed=False)

# -----------------------------
# Data (scaled integers)
# -----------------------------
# f_i (open cost)
f = [5.0, 4.0]

# w_i (per-tech cost)
w = [3.2, 3.0]

# demands d_j (hours)
d = [6.5, 4.0, 7.5]

# H = 8.0 hours per tech
H = 8.0

# assignment costs c_{j,i} (clients rows, centers cols):
c = [
    [1.0, 2.2],
    [2.5,  0.8],
    [1.4, 1.1],
]

# -----------------------------
# Constraints
# -----------------------------
# Binary domains
satx.all_binaries(y)
satx.all_binaries(satx.flatten(x))

# Each client assigned exactly once
for j in range(3):
    assert sum([x[j][i] for i in range(2)]) == 1

# Link: assignment only to open centers
for j in range(3):
    for i in range(2):
        assert x[j][i] <= y[i]

# Integer technician bounds + open-link (M=3)
M = 3
for i in range(2):
    assert s[i] >= 0
    assert s[i] <= M
    assert s[i] <= M * y[i]

# Capacity constraints: sum_j d_j * x_{j,i} <= H * s_i
for i in range(2):
    load_i = sum([d[j] * x[j][i] for j in range(3)])
    assert load_i <= H * s[i]

# At least one center open
assert sum(y) >= 1

# -----------------------------
# Objective (scaled)
# -----------------------------
# Σ f_i y_i + Σ w_i s_i + Σ c_{j,i} x_{j,i}
obj = satx.dot(f, y) + satx.dot(w, s) + sum(c[j][i] * x[j][i] for j in range(3) for i in range(2))

# -----------------------------
# Optimization by iterative tightening
# -----------------------------
optimal = satx.oo()
while satx.satisfy(solver="slime"):
    print("obj =", obj)
    optimal = obj.value
    satx.clear([obj])
    assert obj < optimal
else:
    print("\nFinal best solution")
    print("x =", x)
    print("y =", y)
    print("s =", s)
    print("optimal =", optimal)
    print("\nExpected optimum:")
    print("  y = [0,1], s = [0,3], all clients -> center 2, obj = 17.1")

"""
Output:

SATX The constraint modeling language for SAT solvers
Copyright (c) 2012-2026 Oscar Riveros. all rights reserved.
0.3.9
obj = 32.2
obj = 27.0
obj = 19.5
obj = 17.1

Final best solution
x = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
y = [0.0, 1.0]
s = [0, 3]
optimal = 171/10

Expected optimum:
  y = [0,1], s = [0,3], all clients -> center 2, obj = 17.1

"""