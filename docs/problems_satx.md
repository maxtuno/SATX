---

## 1) Optimización / CP

### 1.1 Knapsack 0/1 (máxima utilidad con restricción de peso)

Patrón: binarios + suma ponderada + maximización por tightening. 

```python
import satx

def knapsack_max(weights, values, capacity, bits=12):
    satx.engine(bits=bits, cnf_path="knapsack.cnf")
    n = len(weights)

    pick = satx.vector(size=n)
    satx.all_binaries(pick)

    wsum = sum(weights[i] * pick[i] for i in range(n))
    vsum = sum(values[i] * pick[i] for i in range(n))

    assert wsum <= capacity

    best_v = None
    best_pick = None

    while satx.satisfy(solver="slime"):
        best_v = vsum.value
        best_pick = [v.value for v in pick]

        satx.clear(pick + [wsum, vsum])
        assert vsum > best_v   # iterative tightening (maximize) :contentReference[oaicite:3]{index=3}

    return best_v, best_pick

if __name__ == "__main__":
    w = [3, 4, 5, 9]
    v = [4, 5, 6, 10]
    cap = 10
    print(knapsack_max(w, v, cap, bits=10))
```

---

### 1.2 Subset Sum / Partition (clásico NP-complete)

SATX ya trae receta directa con `satx.subsets(...)/complement=True`.  

```python
import satx

def subset_sum(universe, target):
    satx.engine(bits=target.bit_length(), cnf_path="subset_sum.cnf")
    bits, subset = satx.subsets(universe)
    assert sum(subset) == target
    assert satx.satisfy(solver="slime")
    return [v.value for v in subset], bits.value

def balanced_partition(data):
    satx.engine(bits=sum(data).bit_length(), cnf_path="partition.cnf")
    bits, sub, com = satx.subsets(data, complement=True)
    assert sum(sub) == sum(com)
    assert satx.satisfy(solver="slime")
    return [v.value for v in sub], [v.value for v in com]

if __name__ == "__main__":
    print(subset_sum([3,5,7,9], 12))
    print(balanced_partition([3,5,7,9]))
```

---

### 1.3 Graph Coloring (k-coloring)

Patrón: variable por nodo ∈ {0..k-1} y desigualdad por arista. (SATX = enteros finitos; evita `or/and` python). 

```python
import satx

def k_coloring(n_nodes, edges, k, bits=4):
    satx.engine(bits=bits, cnf_path="kcolor.cnf")
    c = satx.vector(size=n_nodes)
    for i in range(n_nodes):
        assert 0 <= c[i] < k
    for (u, v) in edges:
        assert c[u] != c[v]
    return c, satx.satisfy(solver="slime")

if __name__ == "__main__":
    n = 5
    edges = [(0,1),(1,2),(2,3),(3,4),(4,0)]  # ciclo impar => k=2 UNSAT
    c, ok = k_coloring(n, edges, k=3, bits=3)
    assert ok
    print([x.value for x in c])
```

---

### 1.4 N-Queens (constraints clásicas)

Patrón: `col[r]` en [0..N-1], `all_different` y diagonales. `satx.gcc` expone `ALLDIFFERENT`. 

```python
import satx
import satx.gcc as gcc

def n_queens(N):
    satx.engine(bits=max(3, (N-1).bit_length()+1), cnf_path="nqueens.cnf")
    col = satx.vector(size=N)
    for r in range(N):
        assert 0 <= col[r] < N

    gcc.all_different(col)  # :contentReference[oaicite:8]{index=8}

    for r1 in range(N):
        for r2 in range(r1 + 1, N):
            dr = r2 - r1
            assert col[r1] != col[r2]              # ya implícito, pero deja claro
            assert col[r1] + dr != col[r2]         # diagonal /
            assert col[r1] - dr != col[r2]         # diagonal \

    assert satx.satisfy(solver="slime")
    return [v.value for v in col]

if __name__ == "__main__":
    print(n_queens(8))
```

---

### 1.5 TSP / Hamiltonian cycle (con `gcc.circuit` o `satx.matrix_permutation`)

`gcc_coverage` confirma `CIRCUIT` implementado. 

Ejemplo minimal: `succ[i]` = sucesor de i; `circuit(succ)` fuerza un ciclo único.

```python
import satx
import satx.gcc as gcc

def tsp_min(cost, bits=10):
    """
    cost: matriz n x n (Python int), con cost[i][i] permitido pero normalmente 0/grande.
    """
    n = len(cost)
    satx.engine(bits=bits, cnf_path="tsp.cnf")

    succ = satx.vector(size=n)
    for i in range(n):
        assert 0 <= succ[i] < n
        assert succ[i] != i

    gcc.circuit(succ)  # un solo ciclo Hamiltoniano :contentReference[oaicite:10]{index=10}

    tour_cost = sum(satx.index(succ[i], cost[i]) for i in range(n))  # tabla por fila :contentReference[oaicite:11]{index=11}

    best = None
    best_succ = None
    while satx.satisfy(solver="slime"):
        best = tour_cost.value
        best_succ = [v.value for v in succ]
        satx.clear(succ + [tour_cost])
        assert tour_cost < best  # minimize via tightening :contentReference[oaicite:12]{index=12}

    return best, best_succ

if __name__ == "__main__":
    cost = [
        [0, 2, 9, 10],
        [1, 0, 6, 4],
        [15,7, 0, 8],
        [6, 3, 12,0],
    ]
    print(tsp_min(cost, bits=8))
```

---

### 1.6 Bin Packing (Global Constraint)

`BIN_PACKING` / `BIN_PACKING_CAPA` están expuestos en `satx.gcc`. 

**Modelo típico:** `bin[i]` asigna ítem i a un contenedor; `gcc.bin_packing_capa(...)` gestiona capacidades.

```python
import satx
import satx.gcc as gcc

def bin_packing(items, capacity, n_bins):
    satx.engine(bits=8, cnf_path="binpack.cnf")
    n = len(items)

    # bin[i] en [0..n_bins-1]
    b = satx.vector(size=n)
    for i in range(n):
        assert 0 <= b[i] < n_bins

    # loads[j] carga del bin j
    loads = satx.vector(size=n_bins)
    for j in range(n_bins):
        assert 0 <= loads[j] <= capacity

    gcc.bin_packing_capa(bins=b, sizes=items, loads=loads, capa=capacity)  # :contentReference[oaicite:14]{index=14}
    assert satx.satisfy(solver="slime")
    return [v.value for v in b], [v.value for v in loads]

if __name__ == "__main__":
    print(bin_packing([4,3,3,2,2], capacity=6, n_bins=3))
```

---

### 1.7 Rectangle Packing / Non-overlap (DIFFN)

`DIFFN` está implementado. 

```python
import satx
import satx.gcc as gcc

def rect_packing(widths, heights, W, H):
    satx.engine(bits=10, cnf_path="diffn.cnf")
    n = len(widths)

    x = satx.vector(size=n)
    y = satx.vector(size=n)

    for i in range(n):
        assert 0 <= x[i] <= W - widths[i]
        assert 0 <= y[i] <= H - heights[i]

    gcc.diffn(x=x, y=y, dx=widths, dy=heights)  # no-overlap :contentReference[oaicite:16]{index=16}
    assert satx.satisfy(solver="slime")
    return [v.value for v in x], [v.value for v in y]

if __name__ == "__main__":
    print(rect_packing([3,2,2], [2,3,2], W=6, H=5))
```

---

## 2) Number Theory / Diophantine

### 2.1 Pythagorean triples: (x^2 + y^2 = r^2)

Receta directa en cookbook (cuida bit-width). 

```python
import satx

def pythagorean(r=10):
    satx.engine(bits=8, cnf_path="pyth.cnf")
    x = satx.integer()
    y = satx.integer()
    assert 0 <= x <= r
    assert 0 <= y <= r
    assert x**2 + y**2 == r*r
    assert satx.satisfy(solver="slime")
    return x.value, y.value

if __name__ == "__main__":
    print(pythagorean(10))
```

---

### 2.2 Factorización pequeña: (p\cdot q = n)

También está en cookbook; útil como subcomponente de modelos más raros. 

```python
import satx

def factor(n):
    satx.engine(bits=max(8, n.bit_length()+1), cnf_path="factor.cnf")
    p = satx.integer()
    q = satx.integer()
    assert p > 1
    assert q > 1
    assert p <= q
    assert p * q == n
    assert satx.satisfy(solver="slime")
    return p.value, q.value

if __name__ == "__main__":
    print(factor(77))
```

---

### 2.3 Exponenciales con exponente variable: (2^k - 7 = x^2)

SATX requiere `engine(..., deep=...)` cuando el exponente es `Unit`.  

```python
import satx

def exp_diophantine():
    satx.engine(bits=10, deep=8, cnf_path="expdio.cnf")
    two = satx.constant(2)
    k = satx.integer()
    x = satx.integer()

    assert 1 <= k <= 8
    assert 0 <= x <= 60
    assert two**k - 7 == x**2

    assert satx.satisfy(solver="slime")
    return k.value, x.value

if __name__ == "__main__":
    print(exp_diophantine())
```

---

### 2.4 GCD como constraint (catálogo)

`gcc.gcd(x, 10, z)` está documentado como wrapper (útil para aritmética modular/filtros). 

```python
import satx
import satx.gcc as gcc

def gcd_example():
    satx.engine(bits=6, cnf_path="gcd.cnf")
    x = satx.integer()
    z = satx.integer()
    gcc.gcd(x, 10, z)  # z = gcd(x,10) :contentReference[oaicite:22]{index=22}
    assert 0 <= x <= 50
    assert z == 5
    assert satx.satisfy(solver="slime")
    return x.value, z.value

if __name__ == "__main__":
    print(gcd_example())
```

---
