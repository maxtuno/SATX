Example: Facility Staffing / Assignment (small MIP)

This document mirrors the data and modeling choices in `example.py`.

---

Problem (plain)

We have 2 candidate service centers and 3 clients.
- Opening center i has a fixed cost f_i.
- Assigning client j to center i has cost c_{j,i}.
- Each client has a (decimal) workload demand d_j hours.
- If center i is opened, we can hire an integer number of technicians s_i (>= 0).
- Each technician provides H = 8.0 hours of capacity.
- Each client must be assigned to exactly one center.
- A client can only be assigned to an opened center.
- Each center's assigned workload must not exceed its capacity.
- At least one center must be opened.

Standard MIP formulation

Sets:
- I = {1,2} (centers)
- J = {1,2,3} (clients)

Parameters (decimals):
- f_i fixed open cost
- w_i cost per technician
- c_{j,i} assignment cost
- d_j demand hours
- H = 8.0 hours per technician

Decision variables:
- y_i in {0,1} open center i
- x_{j,i} in {0,1} assign client j to center i
- s_i in Z_{>=0} technicians hired at center i

Objective:
$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} w_i s_i + \sum_{j \in J} \sum_{i \in I} c_{j,i} x_{j,i}
$$

Constraints:
1) For all $j \in J$:
$$
\sum_{i \in I} x_{j,i} = 1
$$
2) For all $j \in J, i \in I$:
$$
x_{j,i} \le y_i
$$
3) For all $i \in I$:
$$
\sum_{j \in J} d_j x_{j,i} \le H s_i
$$
4) For all $i \in I$:
$$
s_i \le M y_i \quad (M=3)
$$
5) At least one center open:
$$
\sum_{i \in I} y_i \ge 1
$$

---

Fixed-point note (as used in `example.py`)

SATX runs over integers only. This example uses fixed-point scaling by 10
to represent one decimal digit:
- 5.0 -> 50
- 3.2 -> 32
- 6.5 -> 65
- H=8.0 -> 80

The reported objective is also scaled by 10.

In the code:
- `satx.engine(bits=24, fixed_default=True, fixed_scale=10)`
- `y` and `x` are Fixed by default
- `s` is explicit integer with `fixed=False`

---

Data (instance)

Open cost:
- f = [5.0, 4.0]

Per-tech cost:
- w = [3.2, 3.0]

Demand hours:
- d = [6.5, 4.0, 7.5] (total = 18.0)

Assignment cost matrix c (clients rows, centers cols):
- [1.0, 2.2]
- [2.5, 0.8]
- [1.4, 1.1]

---

Known optimum (for comparison)

Open only center 2:
- $y = [0, 1]$
Hire 3 technicians at center 2:
- $s = [0, 3]$
Assign all clients to center 2.

Capacity check:
- load: $6.5 + 4.0 + 7.5 = 18.0$
- capacity: $8.0 \times 3 = 24.0$

Objective value:
- open cost: $4.0$
- tech cost: $3 \times 3.0 = 9.0$
- assignment cost: $2.2 + 0.8 + 1.1 = 4.1$
- total: $17.1$

---

See `example.py` for the runnable SATX model and iterative tightening loop.
