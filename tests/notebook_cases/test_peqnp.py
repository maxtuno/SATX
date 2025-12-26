import functools
import itertools
import math
import operator
from fractions import Fraction

import numpy as np

import satx


def test_peqnp_cell5_exponente_cuadratico():
    satx.engine(32, cnf_path="tests/notebook_cases/tmp_peqnp_cell5.cnf")
    _2 = satx.constant(2)
    n = satx.integer()
    x = satx.integer()
    assert _2**n - 7 == x**2
    assert satx.satisfy(solver="slime")
    assert pow(2, n.value) - 7 == x.value**2


def test_peqnp_cell7_cubica_vs_potencia():
    satx.engine(16, cnf_path="tests/notebook_cases/tmp_peqnp_cell7.cnf")
    _2 = satx.constant(2)
    n = satx.integer()
    k = satx.integer()
    assert n**3 + 10 == _2**k + 5 * n
    assert satx.satisfy(solver="slime")
    assert n.value**3 + 10 == pow(2, k.value) + 5 * n.value


def test_peqnp_cell9_diophantina():
    satx.engine(16, cnf_path="tests/notebook_cases/tmp_peqnp_cell9.cnf")
    a = satx.integer()
    b = satx.integer()
    assert a**2 == b**3 + 1
    assert satx.satisfy(solver="slime")
    assert a.value**2 == b.value**3 + 1


def test_peqnp_cell11_exponencial():
    satx.engine(32, cnf_path="tests/notebook_cases/tmp_peqnp_cell11.cnf")
    _2 = satx.constant(2)
    _3 = satx.constant(3)
    x = satx.integer()
    y = satx.integer()
    assert _3**x == y * _2**x + 1
    assert satx.satisfy(solver="slime")
    assert pow(3, x.value) == y.value * pow(2, x.value) + 1


def test_peqnp_cell13_factorizacion_rsa():
    rsa = 3007
    satx.engine(rsa.bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell13.cnf")
    p = satx.integer()
    q = satx.integer()
    assert p * q == rsa
    assert satx.satisfy(solver="slime")
    assert p.value * q.value == rsa


def test_peqnp_cell15_reverse_copy():
    satx.engine(10, cnf_path="tests/notebook_cases/tmp_peqnp_cell15.cnf")
    x = satx.integer()
    assert x == x.reverse(copy=True)
    assert satx.satisfy(solver="slime")
    assert x.binary == x.binary[::-1]


def test_peqnp_cell17_dot_bias():
    x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_data = [0, 1, 1, 0]
    n, m = len(x_data), len(x_data[0])
    satx.engine(10, cnf_path="tests/notebook_cases/tmp_peqnp_cell17.cnf")
    w = satx.matrix(dimensions=(n, m))
    b = satx.vector(size=n)
    for i in range(n):
        assert y_data[i] == satx.dot(x_data[i], w[i]) + b[i]
    assert satx.satisfy(solver="slime")
    for i in range(n):
        got = sum(x_data[i][j] * w[i][j].value for j in range(m)) + b[i].value
        assert got == y_data[i]


def test_peqnp_cell19_abs_desigualdad():
    satx.engine(4, cnf_path="tests/notebook_cases/tmp_peqnp_cell19.cnf")
    x = satx.integer()
    y = satx.integer()
    assert abs(x - y) == 1
    assert x != satx.oo()
    assert y != satx.oo()
    assert satx.satisfy(solver="slime")
    assert abs(x.value - y.value) == 1


def test_peqnp_cell21_optimizacion_manhattan_minima():
    data = [(0, 0), (1, 0)]
    opt = 1
    while True:
        satx.engine(10, cnf_path="tests/notebook_cases/tmp_peqnp_cell21.cnf")
        x = satx.integer()
        y = satx.integer()
        assert sum(abs(px - x) + abs(py - y) for px, py in data) < opt
        assert x != satx.oo()
        assert y != satx.oo()
        if satx.satisfy(solver="slime"):
            break
        opt += 1
    assert opt == 2
    assert sum(abs(px - x.value) + abs(py - y.value) for px, py in data) < opt


def test_peqnp_cell23_diferencia_de_cuadrados():
    rsa = 3007
    satx.engine(rsa.bit_length() + 1, cnf_path="tests/notebook_cases/tmp_peqnp_cell23.cnf")
    p = satx.integer()
    q = satx.integer()
    assert p**2 - q**2 == rsa
    assert q < p
    assert satx.satisfy(solver="slime")
    assert p.value**2 - q.value**2 == rsa
    assert q.value < p.value


def test_peqnp_cell25_potencia_general():
    satx.engine(10, cnf_path="tests/notebook_cases/tmp_peqnp_cell25.cnf")
    x = satx.integer()
    y = satx.integer()
    z = satx.integer()
    assert x**y == z
    assert satx.satisfy(solver="slime")
    assert pow(x.value, y.value) == z.value


def test_peqnp_cell27_x2_mas_c_igual_3n():
    n_bits = 32
    satx.engine(n_bits.bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell27.cnf")
    _3 = satx.constant(3)
    n = satx.integer()
    x = satx.integer()
    c = satx.integer()
    assert x**2 + c == _3**n
    assert x > 1
    assert c > 1
    assert n > 1
    assert satx.satisfy(solver="slime")
    assert x.value**2 + c.value == pow(3, n.value)
    assert x.value > 1
    assert c.value > 1
    assert n.value > 1


def test_peqnp_cell29_factorial():
    satx.engine(32, cnf_path="tests/notebook_cases/tmp_peqnp_cell29.cnf")
    x = satx.integer()
    satx.factorial(x) == math.factorial(10)
    assert satx.satisfy(solver="slime")
    assert math.factorial(x.value) == math.factorial(10)


def test_peqnp_cell31_sigma_suma_cuadrados():
    satx.engine(16, cnf_path="tests/notebook_cases/tmp_peqnp_cell31.cnf")
    x = satx.integer()
    n = satx.integer()
    satx.sigma(lambda k: k**2, 1, n) == x
    assert satx.satisfy(solver="slime")
    assert x.value == sum(k**2 for k in range(1, n.value + 1))


def test_peqnp_cell33_pi_producto_cuadrados():
    satx.engine(32, cnf_path="tests/notebook_cases/tmp_peqnp_cell33.cnf")
    x = satx.integer()
    n = satx.integer()
    satx.pi(lambda k: k**2, 1, n) == x
    assert 0 < x <= 2 ** math.log(satx.oo())
    assert n > 0
    assert satx.satisfy(solver="slime")
    prod = functools.reduce(operator.mul, (k**2 for k in range(1, n.value + 1)), 1)
    assert x.value == prod


def test_peqnp_cell35_fibonacci():
    n = 10
    satx.engine(n, cnf_path="tests/notebook_cases/tmp_peqnp_cell35.cnf")
    x = satx.vector(size=n + 1)
    assert x[0] == 0
    assert x[1] == 1
    for i in range(2, n + 1):
        assert x[i - 1] + x[i - 2] == x[i]
    assert satx.satisfy(solver="slime")
    xs = [v.value for v in x]
    for i in range(2, n + 1):
        assert xs[i] == xs[i - 1] + xs[i - 2]
    assert xs[n] == 55


def test_peqnp_cell37_tensor_suma_y_bits():
    satx.engine(10, cnf_path="tests/notebook_cases/tmp_peqnp_cell37.cnf")
    x = satx.tensor(dimensions=(4,))
    y = satx.tensor(dimensions=(2, 2))
    assert x + y == 10
    assert x[[0]](0, 1) == 1
    assert y[[0, 0]](0, 1) == 1
    assert satx.satisfy(solver="slime")
    assert x.value + y.value == 10
    assert x.value % 2 == 1
    assert y.value % 2 == 1


def test_peqnp_cell38_tensor_condicional_sum_zero():
    n = 2
    satx.engine(4, cnf_path="tests/notebook_cases/tmp_peqnp_cell38.cnf")
    x = satx.tensor(dimensions=(n, n))
    a = satx.integer()
    b = satx.integer()
    assert sum(x[[i, j]](a**2 - b**3, a**3 - b**2) for i in range(n) for j in range(n)) == 0
    assert satx.satisfy(solver="slime")
    left = a.value**2 - b.value**3
    right = a.value**3 - b.value**2
    s = 0
    for i in range(n):
        for j in range(n):
            s += left if not x.binary[i][j] else right
    assert s == 0


def test_peqnp_cell40_factorizacion_tensores_rsa():
    rsa = 3007
    satx.engine(rsa.bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell40.cnf")
    p = satx.tensor(dimensions=(satx.bits(),))
    q = satx.tensor(dimensions=(satx.bits(),))
    assert p * q == rsa
    assert p[[0]](0, 1) == 1
    assert q[[0]](0, 1) == 1
    assert sum(p[[i]](0, 1) for i in range(satx.bits() // 2 + 1, satx.bits())) == 0
    assert sum(q[[i]](0, 1) for i in range(satx.bits() // 2, satx.bits())) == 0
    assert satx.satisfy(solver="slime")
    assert p.value * q.value == rsa
    assert p.value % 2 == 1
    assert q.value % 2 == 1
    assert p.value < (1 << (satx.bits() // 2 + 1))
    assert q.value < (1 << (satx.bits() // 2))


def _sat_assignment_satisfies(sat, assignment_bits):
    for clause in sat:
        ok = False
        for lit in clause:
            var = abs(lit) - 1
            val = assignment_bits[var]
            ok |= (val if lit > 0 else not val)
            if ok:
                break
        if not ok:
            return False
    return True


def test_peqnp_cell42_sat_instance_main_guard():
    n, sat = 10, [
        [9, -5, 10, -6, 3],
        [6, 8],
        [8, 4],
        [-10, 5],
        [-9, 8],
        [-9, -3],
        [-2, 5],
        [6, 4],
        [-2, -1],
        [7, -2],
        [-9, 4],
        [-1, -10],
        [-3, 4],
        [7, 5],
        [6, -3],
        [-10, 7],
        [-1, 7],
        [8, -3],
        [-2, -10],
        [-1, 5],
        [-7, 1, 9, -6, 3],
        [-9, 6],
        [-8, 10, -5, -4, 2],
        [-4, -7, 1, -8, 2],
    ]
    satx.engine(bits=1, cnf_path="tests/notebook_cases/tmp_peqnp_cell42.cnf")
    x = satx.tensor(dimensions=(n,))
    assert functools.reduce(
        operator.iand,
        (functools.reduce(operator.ior, (x[[abs(lit) - 1]](lit < 0, lit > 0) for lit in cls)) for cls in sat),
    ) == 1
    assert satx.satisfy(solver="slime")
    assert _sat_assignment_satisfies(sat, list(map(bool, x.binary)))


def test_peqnp_cell44_subset_sum_tensor():
    universe = [3, 5, 7, 9]
    t = 12
    satx.engine(t.bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell44.cnf")
    x = satx.tensor(dimensions=(len(universe),))
    assert sum(x[[i]](0, universe[i]) for i in range(len(universe))) == t
    assert satx.satisfy(solver="slime")
    picked = [universe[i] for i in range(len(universe)) if x.binary[i]]
    assert sum(picked) == t


def test_peqnp_cell46_diferencias_con_tip():
    original = [1, 3, 6, 10, 15]
    diffs = [abs(original[i] - original[i - 1]) for i in range(1, len(original))]
    ith = 2
    tip = original[ith]
    n = len(original)
    satx.engine(sum(diffs).bit_length() + 4, cnf_path="tests/notebook_cases/tmp_peqnp_cell46.cnf")
    x = satx.vector(size=n)
    assert tip == satx.index(ith, x)
    for i in range(n - 1):
        assert x[i] <= x[i + 1]
        assert satx.index(i, diffs) == x[i + 1] - x[i]
    assert satx.satisfy(solver="slime")
    xs = [v.value for v in x]
    assert xs == original
    assert [abs(xs[i + 1] - xs[i]) for i in range(n - 1)] == diffs


def test_peqnp_cell49_cubica_vs_cuadratica():
    satx.engine(10, cnf_path="tests/notebook_cases/tmp_peqnp_cell49.cnf")
    x = satx.integer()
    y = satx.integer()
    assert x**3 - x + 1 == y**2
    assert x != 0
    assert y != 0
    assert satx.satisfy(solver="slime")
    assert x.value**3 - x.value + 1 == y.value**2
    assert x.value != 0
    assert y.value != 0


def test_peqnp_cell51_racionales_ecuacion():
    satx.engine(10, cnf_path="tests/notebook_cases/tmp_peqnp_cell51.cnf")
    x = satx.rational()
    y = satx.rational()
    assert x**3 + x * y == y**2
    assert x != 0
    assert y != 0
    assert satx.satisfy(solver="slime")
    xf = Fraction(int(x.numerator), int(x.denominator))
    yf = Fraction(int(y.numerator), int(y.denominator))
    assert xf != 0
    assert yf != 0
    assert xf**3 + xf * yf == yf**2


def test_peqnp_cell53_norma_racional():
    dim = 2
    satx.engine(5, cnf_path="tests/notebook_cases/tmp_peqnp_cell53.cnf")
    ps = satx.vector(size=dim, is_rational=True)
    assert sum([p**dim for p in ps]) <= 1
    assert satx.satisfy(solver="slime")
    vals = [Fraction(int(p.numerator), int(p.denominator)) for p in ps]
    assert sum(v**dim for v in vals) <= 1


def test_peqnp_cell56_sat_instance_sin_guard():
    n, sat = 10, [
        [9, -5, 10, -6, 3],
        [6, 8],
        [8, 4],
        [-10, 5],
        [-9, 8],
        [-9, -3],
        [-2, 5],
        [6, 4],
        [-2, -1],
        [7, -2],
        [-9, 4],
        [-1, -10],
        [-3, 4],
        [7, 5],
        [6, -3],
        [-10, 7],
        [-1, 7],
        [8, -3],
        [-2, -10],
        [-1, 5],
        [-7, 1, 9, -6, 3],
        [-9, 6],
        [-8, 10, -5, -4, 2],
        [-4, -7, 1, -8, 2],
    ]
    satx.engine(bits=1, cnf_path="tests/notebook_cases/tmp_peqnp_cell56.cnf")
    x = satx.tensor(dimensions=(n,))
    assert functools.reduce(
        operator.iand,
        (functools.reduce(operator.ior, (x[[abs(lit) - 1]](lit < 0, lit > 0) for lit in cls)) for cls in sat),
    ) == 1
    assert satx.satisfy(solver="slime")
    assert _sat_assignment_satisfies(sat, list(map(bool, x.binary)))


def test_peqnp_cell58_clique():
    k = 3
    n = 5
    edges = {(1, 0), (0, 2), (1, 4), (2, 1), (4, 2), (3, 2)}
    satx.engine(bits=k.bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell58.cnf")
    bits = satx.integer(bits=n)
    assert sum(satx.switch(bits, i) for i in range(n)) == k
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (i, j) not in edges and (j, i) not in edges:
                assert satx.switch(bits, i) + satx.switch(bits, j) <= 1
    assert satx.satisfy(solver="slime")
    chosen = [i for i in range(n) if not bits.binary[i]]
    assert len(chosen) == k
    for i, j in itertools.combinations(chosen, 2):
        assert (i, j) in edges or (j, i) in edges


def test_peqnp_cell60_vertex_cover():
    n = 5
    graph = [(1, 0), (0, 2), (1, 4), (2, 1), (4, 2), (3, 2)]
    vertex = list(range(n))
    k = 3
    satx.engine(bits=n.bit_length() + 1, cnf_path="tests/notebook_cases/tmp_peqnp_cell60.cnf")
    index = satx.integer(bits=n)
    for i, j in graph:
        assert satx.switch(index, vertex.index(i), neg=True) + satx.switch(index, vertex.index(j), neg=True) >= 1
    assert sum(satx.switch(index, vertex.index(i), neg=True) for i in vertex) == k
    assert satx.satisfy(solver="slime")
    cover = [vertex[i] for i in range(n) if index.binary[i]]
    assert len(cover) == k
    for i, j in graph:
        assert i in cover or j in cover


def test_peqnp_cell62_cubo_latino_restricciones():
    n = 3
    m = 3
    satx.engine(n.bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell62.cnf")
    Y = satx.vector(size=n**m)
    satx.apply_single(Y, lambda k: k < n)
    Y = np.reshape(Y, m * [n])
    for axis in range(m):
        for fixed in itertools.product(range(n), repeat=m - 1):
            idx = list(fixed)
            idx.insert(axis, slice(None))
            line = np.asarray(Y[tuple(idx)]).tolist()
            satx.all_different(line)
    assert satx.satisfy(solver="slime")
    y = np.vectorize(int)(Y)
    assert ((0 <= y) & (y < n)).all()
    for axis in range(m):
        for fixed in itertools.product(range(n), repeat=m - 1):
            idx = list(fixed)
            idx.insert(axis, slice(None))
            line = y[tuple(idx)].tolist()
            assert len(set(line)) == n


def test_peqnp_cell65_tsp_optimizacion():
    n = 4
    matrix = np.asarray(
        [
            [0, 1, 4, 3],
            [1, 0, 2, 5],
            [4, 2, 0, 1],
            [3, 5, 1, 0],
        ],
        dtype=int,
    )
    flat = matrix.flatten().tolist()
    best = sum(flat) + 1
    while True:
        satx.engine(int(sum(flat)).bit_length() + 1, cnf_path="tests/notebook_cases/tmp_peqnp_cell65.cnf")
        x, y = satx.matrix_permutation(flat, n)
        assert sum(y) < best
        if not satx.satisfy(solver="slime"):
            break
        best = sum(y)
        satx.clear(x)
        satx.clear(y)
    # brute-force optimum (cycle over permutations)
    brute = math.inf
    for perm in itertools.permutations(range(n)):
        s = 0
        for i in range(n):
            s += matrix[perm[i]][perm[(i + 1) % n]]
        brute = min(brute, s)
    assert best == brute


def test_peqnp_cell67_cuadrado_magico_3x3():
    n = 3
    satx.engine(5, cnf_path="tests/notebook_cases/tmp_peqnp_cell67.cnf")
    c = satx.integer()
    xs = satx.matrix(dimensions=(n, n))
    satx.apply_single(satx.flatten(xs), lambda x: x > 0)
    satx.all_different(satx.flatten(xs))
    for i in range(n):
        assert sum(xs[i][j] for j in range(n)) == c
    for j in range(n):
        assert sum(xs[i][j] for i in range(n)) == c
    assert sum(xs[i][i] for i in range(n)) == c
    assert sum(xs[i][n - 1 - i] for i in range(n)) == c
    assert satx.satisfy(solver="slime")
    grid = np.vectorize(int)(xs)
    assert (grid > 0).all()
    assert len(set(grid.flatten().tolist())) == n * n
    target = int(c)
    assert all(int(sum(grid[i, :])) == target for i in range(n))
    assert all(int(sum(grid[:, j])) == target for j in range(n))
    assert int(sum(grid[i, i] for i in range(n))) == target
    assert int(sum(grid[i, n - 1 - i] for i in range(n))) == target


def test_peqnp_cell69_tripletas_permutacion():
    triplets = [1, 2, 3, 4, 5, 9]
    size = len(triplets)
    satx.engine(bits=max(triplets).bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell69.cnf")
    xs, ys = satx.permutations(triplets, size)
    for i in range(0, size, 3):
        assert ys[i] + ys[i + 1] == ys[i + 2]
    assert satx.satisfy(solver="slime")
    yv = [v.value for v in ys]
    assert sorted(yv) == sorted(triplets)
    for i in range(0, size, 3):
        assert yv[i] + yv[i + 1] == yv[i + 2]


def test_peqnp_cell71_subset_sum_subsets():
    universe = [3, 5, 7, 9]
    t = 12
    satx.engine(t.bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell71.cnf")
    bits, subset = satx.subsets(universe)
    assert sum(subset) == t
    assert satx.satisfy(solver="slime")
    subset_values = [x.value for x in subset]
    assert sum(subset_values) == t
    assert all(v in (0, universe[i]) for i, v in enumerate(subset_values))
    chosen = [universe[i] for i, v in enumerate(subset_values) if v == universe[i]]
    assert sum(chosen) == t


def test_peqnp_cell73_differences_one_of():
    n = 6
    diffs = [2, 3, 4, 3, 1]
    satx.engine(n.bit_length() + 1, cnf_path="tests/notebook_cases/tmp_peqnp_cell73.cnf")
    x = satx.vector(size=n)
    satx.all_different(x)
    satx.apply_single(x, lambda a: 1 <= a <= n)
    for i in range(n - 1):
        assert satx.index(i, diffs) == satx.one_of([x[i + 1] - x[i], x[i] - x[i + 1]])
    assert satx.satisfy(solver="slime")
    xs = [v.value for v in x]
    assert sorted(xs) == list(range(1, n + 1))
    assert [abs(xs[i + 1] - xs[i]) for i in range(n - 1)] == diffs


def test_peqnp_cell75_hamiltoniano_con_matriz():
    n = 4
    M = np.zeros((n, n), dtype=int)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        M[a][b] = 1
    satx.engine((n**2).bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell75.cnf")
    ids, elements = satx.matrix_permutation((1 - M).flatten().tolist(), n)
    assert sum(elements) == 0
    assert satx.satisfy(solver="slime")
    perm = [i.value for i in ids]
    assert sorted(perm) == list(range(n))
    for i in range(n):
        assert M[perm[i]][perm[(i + 1) % n]] == 1


def test_peqnp_cell77_bin_packing():
    capacity = 10
    elements = [6, 4, 3, 2]
    bins = 2
    satx.engine(bits=capacity.bit_length() + 1, cnf_path="tests/notebook_cases/tmp_peqnp_cell77.cnf")
    slots = satx.vector(bits=len(elements), size=bins)
    for i in range(len(elements)):
        assert sum(satx.switch(slot, i) for slot in slots) == 1
    for slot in slots:
        assert sum(satx.switch(slot, i) * elements[i] for i in range(len(elements))) <= capacity
    assert satx.satisfy(solver="slime")
    assigned_counts = [0] * len(elements)
    for slot in slots:
        picked = [i for i in range(len(elements)) if not slot.binary[i]]
        for i in picked:
            assigned_counts[i] += 1
        assert sum(elements[i] for i in picked) <= capacity
    assert assigned_counts == [1] * len(elements)


def test_peqnp_cell79_sistema_lineal_binario():
    cc = np.asarray([[2, 3, 5], [7, 11, 13], [1, 0, 4]], dtype=int)
    expected = np.asarray([1, 0, 1], dtype=int)
    d = np.dot(cc, expected)
    satx.engine(bits=int(np.sum(cc)).bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell79.cnf")
    xs = satx.vector(size=cc.shape[1])
    satx.all_binaries(xs)
    assert (np.dot(cc, xs) == d).all()
    assert satx.satisfy(solver="slime")
    got = np.asarray([v.value for v in xs], dtype=int)
    assert (np.dot(cc, got) == d).all()


def test_peqnp_cell81_n_queens_completion():
    n = 8
    placed = [(0, 0), (1, 4), (2, 7)]
    satx.engine(bits=n.bit_length() + 1, cnf_path="tests/notebook_cases/tmp_peqnp_cell81.cnf")
    qs = satx.vector(size=n)
    for r, c in placed:
        assert qs[r] == c
    satx.apply_single(qs, lambda x: x < n)
    satx.apply_dual(qs, lambda x, y: x != y)
    satx.apply_dual([qs[i] + i for i in range(n)], lambda x, y: x != y)
    satx.apply_dual([qs[i] - i for i in range(n)], lambda x, y: x != y)
    assert satx.satisfy(solver="slime")
    cols = [q.value for q in qs]
    assert sorted(cols) == list(range(n))
    for r1 in range(n):
        for r2 in range(r1 + 1, n):
            assert cols[r1] != cols[r2]
            assert abs(cols[r1] - cols[r2]) != abs(r1 - r2)


def test_peqnp_cell83_partition_equal_sum():
    data = [3, 5, 7, 9]
    satx.engine(int(sum(data)).bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell83.cnf")
    T, sub, com = satx.subsets(data, complement=True)
    assert sum(sub) == sum(com)
    assert satx.satisfy(solver="slime")
    sub_vals = [v.value for v in sub]
    com_vals = [v.value for v in com]
    assert sum(sub_vals) == sum(com_vals)
    assert all(v in (0, data[i]) for i, v in enumerate(sub_vals))
    assert all(v in (0, data[i]) for i, v in enumerate(com_vals))


def test_peqnp_cell85_sudoku_base2():
    base = 2
    side = base * base
    puzzle = np.asarray(
        [
            [1, 0, 0, 4],
            [0, 0, 1, 0],
            [0, 4, 0, 0],
            [2, 0, 0, 3],
        ],
        dtype=int,
    )
    satx.engine(side.bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell85.cnf")
    board = np.asarray(satx.matrix(dimensions=(side, side)))
    satx.apply_single(board.flatten(), lambda x: 1 <= x <= side)
    for i in range(side):
        for j in range(side):
            if puzzle[i][j]:
                assert board[i][j] == puzzle[i][j]
    for c, r in zip(board, board.T):
        satx.all_different(c.tolist())
        satx.all_different(r.tolist())
    for i in range(base):
        for j in range(base):
            block = board[i * base : (i + 1) * base, j * base : (j + 1) * base].flatten().tolist()
            satx.all_different(block)
    assert satx.satisfy(solver="slime")
    solved = np.vectorize(int)(board)
    for i in range(side):
        assert set(solved[i, :].tolist()) == set(range(1, side + 1))
        assert set(solved[:, i].tolist()) == set(range(1, side + 1))
    for i in range(base):
        for j in range(base):
            block = solved[i * base : (i + 1) * base, j * base : (j + 1) * base].flatten().tolist()
            assert set(block) == set(range(1, side + 1))


def test_peqnp_cell87_partition_balanceada_con_k():
    D = [1, 2, 3, 4, 5, 9]
    k = len(D) // 2
    satx.engine(sum(D).bit_length(), cnf_path="tests/notebook_cases/tmp_peqnp_cell87.cnf")
    bins, sub, com = satx.subsets(D, k, complement=True)
    assert sum(sub) == sum(com)
    assert satx.satisfy(solver="slime")
    sub_vals = [v.value for v in sub]
    com_vals = [v.value for v in com]
    assert sum(sub_vals) == sum(com_vals)
    assert sum(1 for v in sub_vals if v != 0) == k
    assert sum(1 for v in com_vals if v != 0) == k
