import satx


def test_sat_x_cell4_inequaciones_simples():
    satx.engine(bits=10, cnf_path="tests/notebook_cases/tmp_sat_x_cell4.cnf")
    x = satx.integer()
    y = satx.integer()
    assert 0 < x <= 3
    assert 0 < y <= 3
    assert x + y > 2
    assert satx.satisfy(solver="slime")
    assert 0 < x.value <= 3
    assert 0 < y.value <= 3
    assert x.value + y.value > 2


def test_sat_x_cell9_circulo_pitagorico():
    satx.engine(bits=10, cnf_path="tests/notebook_cases/tmp_sat_x_cell9.cnf")
    x = satx.integer()
    y = satx.integer()
    assert x**2 + y**2 == 100
    assert satx.satisfy(solver="slime")
    assert x.value**2 + y.value**2 == 100


def test_sat_x_cell11_subset_sum():
    universe = [3, 5, 7, 9]
    target = 12
    satx.engine(target.bit_length(), cnf_path="tests/notebook_cases/tmp_sat_x_cell11.cnf")
    idx, subset, complement = satx.subsets(universe, complement=True)
    assert sum(subset) == target
    assert satx.satisfy(solver="slime")
    subset_values = [x.value for x in subset]
    complement_values = [x.value for x in complement]
    assert sum(subset_values) == target
    assert sum(complement_values) == sum(universe) - target
    assert all(v in (0, universe[i]) for i, v in enumerate(subset_values))
    assert all(v in (0, universe[i]) for i, v in enumerate(complement_values))
    chosen = [universe[i] for i in range(len(universe)) if not idx.binary[i]]
    assert sum(chosen) == target


def _tensor_factorization_case(cnf_path: str):
    rsa = 3007
    satx.engine(rsa.bit_length(), cnf_path=cnf_path)
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


def test_sat_x_cell13_factorizacion_tensores():
    _tensor_factorization_case("tests/notebook_cases/tmp_sat_x_cell13.cnf")


def test_sat_x_cell14_factorizacion_tensores():
    _tensor_factorization_case("tests/notebook_cases/tmp_sat_x_cell14.cnf")


def test_sat_x_cell15_factorizacion_tensores():
    _tensor_factorization_case("tests/notebook_cases/tmp_sat_x_cell15.cnf")


def _differences_one_of_case(cnf_path: str):
    n = 6
    diffs = [2, 3, 4, 3, 1]
    satx.engine(n.bit_length() + 1, cnf_path=cnf_path)
    x = satx.vector(size=n)
    satx.all_different(x)
    satx.apply_single(x, lambda a: 1 <= a <= n)
    for i in range(n - 1):
        assert satx.index(i, diffs) == satx.one_of([x[i + 1] - x[i], x[i] - x[i + 1]])
    assert satx.satisfy(solver="slime")
    xs = [v.value for v in x]
    assert sorted(xs) == list(range(1, n + 1))
    assert [abs(xs[i + 1] - xs[i]) for i in range(n - 1)] == diffs


def test_sat_x_cell17_differences_one_of():
    _differences_one_of_case("tests/notebook_cases/tmp_sat_x_cell17.cnf")


def test_sat_x_cell18_differences_one_of():
    _differences_one_of_case("tests/notebook_cases/tmp_sat_x_cell18.cnf")


def test_sat_x_cell19_differences_one_of():
    _differences_one_of_case("tests/notebook_cases/tmp_sat_x_cell19.cnf")


def test_sat_x_cell21_bin_packing():
    capacity = 10
    elements = [6, 4, 3, 2]
    bins = 2
    satx.engine(bits=capacity.bit_length() + 1, cnf_path="tests/notebook_cases/tmp_sat_x_cell21.cnf")
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
