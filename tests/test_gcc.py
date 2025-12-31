"""
Copyright (c) 2012-2026 Oscar Riveros

SATX is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

SATX is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Commercial licensing options are available.
See COMMERCIAL.md for details.
"""
import satx
import satx.gcc as gcc


def test_all_equal_except_0_sat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_all_equal_except_0_sat.cnf")
    x, y, z = satx.vector(size=3)
    assert x == 0
    assert y == 5
    assert z == 5
    gcc.all_equal_except_0([x, y, z])
    assert satx.satisfy(solver="slime")
    vals = [x.value, y.value, z.value]
    nonzero = {v for v in vals if v != 0}
    assert nonzero == {5}


def test_all_equal_except_0_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_all_equal_except_0_unsat.cnf")
    x, y, z = satx.vector(size=3)
    assert x == 1
    assert y == 2
    assert z == 0
    gcc.all_equal_except_0([x, y, z])
    assert not satx.satisfy(solver="slime")


def test_in_set_sat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_in_set_sat.cnf")
    x = satx.integer()
    gcc.in_(x, [1, 3, 5])
    assert satx.satisfy(solver="slime")
    assert x.value in {1, 3, 5}


def test_in_set_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_in_set_unsat.cnf")
    x = satx.integer()
    gcc.in_(x, [1, 3])
    assert x == 2
    assert not satx.satisfy(solver="slime")


def test_not_in_sat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_not_in_sat.cnf")
    x = satx.integer()
    assert x == 4
    gcc.not_in(x, [1, 2, 3])
    assert satx.satisfy(solver="slime")
    assert x.value not in {1, 2, 3}


def test_not_in_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_not_in_unsat.cnf")
    x = satx.integer()
    assert x == 2
    gcc.not_in(x, [1, 2, 3])
    assert not satx.satisfy(solver="slime")


def test_among_sat():
    satx.engine(bits=5, cnf_path="tests/tmp_gcc_among_sat.cnf")
    x, y, z = satx.vector(size=3)
    assert x == 1
    assert y == 2
    assert z == 7
    gcc.among(2, [x, y, z], [1, 2])
    assert satx.satisfy(solver="slime")
    assert sum(v in {1, 2} for v in [x.value, y.value, z.value]) == 2


def test_among_unsat():
    satx.engine(bits=5, cnf_path="tests/tmp_gcc_among_unsat.cnf")
    x, y, z = satx.vector(size=3)
    assert x == 1
    assert y == 2
    assert z == 1
    gcc.among(1, [x, y, z], [1, 2])
    assert not satx.satisfy(solver="slime")


def test_among_var_sat():
    satx.engine(bits=5, cnf_path="tests/tmp_gcc_among_var_sat.cnf")
    x, y, z = satx.vector(size=3)
    n = satx.integer()
    assert x == 1
    assert y == 7
    assert z == 2
    gcc.among_var(n, [x, y, z], [1, 2])
    assert satx.satisfy(solver="slime")
    assert n.value == 2


def test_among_var_unsat():
    satx.engine(bits=5, cnf_path="tests/tmp_gcc_among_var_unsat.cnf")
    x, y, z = satx.vector(size=3)
    n = satx.integer()
    assert x == 1
    assert y == 7
    assert z == 2
    gcc.among_var(n, [x, y, z], [1, 2])
    assert n == 1
    assert not satx.satisfy(solver="slime")


def test_at_least_sat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_at_least_sat.cnf")
    x, y, z = satx.vector(size=3)
    assert x == 1
    assert y == 1
    gcc.at_least(2, [x, y, z], 1)
    assert satx.satisfy(solver="slime")
    assert sum(v == 1 for v in [x.value, y.value, z.value]) >= 2


def test_at_least_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_at_least_unsat.cnf")
    x, y, z = satx.vector(size=3)
    assert x == 1
    assert y == 1
    assert z != 1
    gcc.at_least(3, [x, y, z], 1)
    assert not satx.satisfy(solver="slime")


def test_at_most_sat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_at_most_sat.cnf")
    x, y, z = satx.vector(size=3)
    assert x == 1
    assert y == 2
    assert z == 3
    gcc.at_most(1, [x, y, z], 1)
    assert satx.satisfy(solver="slime")
    assert sum(v == 1 for v in [x.value, y.value, z.value]) <= 1


def test_at_most_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_at_most_unsat.cnf")
    x, y, z = satx.vector(size=3)
    assert x == 1
    assert y == 1
    gcc.at_most(1, [x, y, z], 1)
    assert not satx.satisfy(solver="slime")


def test_exactly_sat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_exactly_sat.cnf")
    x, y, z = satx.vector(size=3)
    assert x == 1
    assert y == 1
    assert z == 2
    gcc.exactly(2, [x, y, z], 1)
    assert satx.satisfy(solver="slime")
    assert sum(v == 1 for v in [x.value, y.value, z.value]) == 2


def test_exactly_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_exactly_unsat.cnf")
    x, y, z = satx.vector(size=3)
    assert x == 1
    assert y == 2
    assert z == 3
    gcc.exactly(2, [x, y, z], 1)
    assert not satx.satisfy(solver="slime")


def test_minimum_sat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_minimum_sat.cnf")
    x, y, z = satx.vector(size=3)
    mn = satx.integer()
    assert x == 3
    assert y == 5
    assert z == 4
    gcc.minimum(mn, [x, y, z])
    assert satx.satisfy(solver="slime")
    assert mn.value == min([x.value, y.value, z.value])


def test_minimum_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_minimum_unsat.cnf")
    x, y, z = satx.vector(size=3)
    mn = satx.integer()
    assert x == 3
    assert y == 5
    assert z == 4
    gcc.minimum(mn, [x, y, z])
    assert mn == 4
    assert not satx.satisfy(solver="slime")


def test_maximum_sat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_maximum_sat.cnf")
    x, y, z = satx.vector(size=3)
    mx = satx.integer()
    assert x == 3
    assert y == 5
    assert z == 4
    gcc.maximum(mx, [x, y, z])
    assert satx.satisfy(solver="slime")
    assert mx.value == max([x.value, y.value, z.value])


def test_maximum_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_maximum_unsat.cnf")
    x, y, z = satx.vector(size=3)
    mx = satx.integer()
    assert x == 3
    assert y == 5
    assert z == 4
    gcc.maximum(mx, [x, y, z])
    assert mx == 4
    assert not satx.satisfy(solver="slime")


def test_sum_sat():
    satx.engine(bits=5, cnf_path="tests/tmp_gcc_sum_sat.cnf")
    x, y = satx.vector(size=2)
    assert x == 2
    assert y == 3
    gcc.sum([x, y], rel=lambda a, b: a == b, rhs=5)
    assert satx.satisfy(solver="slime")
    assert x.value + y.value == 5


def test_sum_unsat():
    satx.engine(bits=5, cnf_path="tests/tmp_gcc_sum_unsat.cnf")
    x, y = satx.vector(size=2)
    assert x == 2
    assert y == 3
    gcc.sum([x, y], rel=lambda a, b: a == b, rhs=6)
    assert not satx.satisfy(solver="slime")


def test_scalar_product_sat():
    satx.engine(bits=6, cnf_path="tests/tmp_gcc_scalar_product_sat.cnf")
    x, y = satx.vector(size=2)
    assert x == 1
    assert y == 2
    gcc.scalar_product([2, 3], [x, y], rel=lambda a, b: a == b, rhs=8)
    assert satx.satisfy(solver="slime")
    assert 2 * x.value + 3 * y.value == 8


def test_scalar_product_unsat():
    satx.engine(bits=6, cnf_path="tests/tmp_gcc_scalar_product_unsat.cnf")
    x, y = satx.vector(size=2)
    assert x == 1
    assert y == 2
    gcc.scalar_product([2, 3], [x, y], rel=lambda a, b: a == b, rhs=7)
    assert not satx.satisfy(solver="slime")


def test_nvalue_sat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_nvalue_sat.cnf")
    xs = satx.vector(size=4)
    n = satx.integer()
    assert xs[0] == 1
    assert xs[1] == 1
    assert xs[2] == 2
    assert xs[3] == 3
    gcc.nvalue(n, xs)
    assert satx.satisfy(solver="slime")
    assert n.value == len(set(v.value for v in xs))


def test_nvalue_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_nvalue_unsat.cnf")
    xs = satx.vector(size=4)
    n = satx.integer()
    assert xs[0] == 1
    assert xs[1] == 1
    assert xs[2] == 2
    assert xs[3] == 3
    gcc.nvalue(n, xs)
    assert n == 2
    assert not satx.satisfy(solver="slime")


def test_lex_less_sat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_lex_less_sat.cnf")
    a = satx.vector(size=2)
    b = satx.vector(size=2)
    assert a[0] == 1
    assert a[1] == 2
    assert b[0] == 1
    assert b[1] == 3
    gcc.lex_less(a, b)
    assert satx.satisfy(solver="slime")
    assert tuple(v.value for v in a) < tuple(v.value for v in b)


def test_lex_less_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_lex_less_unsat.cnf")
    a = satx.vector(size=2)
    b = satx.vector(size=2)
    assert a[0] == 1
    assert a[1] == 3
    assert b[0] == 1
    assert b[1] == 2
    gcc.lex_less(a, b)
    assert not satx.satisfy(solver="slime")


def test_lex_lesseq_sat_equal():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_lex_lesseq_sat.cnf")
    a = satx.vector(size=2)
    b = satx.vector(size=2)
    assert a[0] == 1
    assert a[1] == 2
    assert b[0] == 1
    assert b[1] == 2
    gcc.lex_lesseq(a, b)
    assert satx.satisfy(solver="slime")
    assert tuple(v.value for v in a) <= tuple(v.value for v in b)


def test_lex_lesseq_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_lex_lesseq_unsat.cnf")
    a = satx.vector(size=2)
    b = satx.vector(size=2)
    assert a[0] == 2
    assert a[1] == 0
    assert b[0] == 1
    assert b[1] == 9
    gcc.lex_lesseq(a, b)
    assert not satx.satisfy(solver="slime")


def test_lex_greater_sat():
    satx.engine(bits=5, cnf_path="tests/tmp_gcc_lex_greater_sat.cnf")
    a = satx.vector(size=2)
    b = satx.vector(size=2)
    assert a[0] == 2
    assert a[1] == 0
    assert b[0] == 1
    assert b[1] == 9
    gcc.lex_greater(a, b)
    assert satx.satisfy(solver="slime")
    assert tuple(v.value for v in a) > tuple(v.value for v in b)


def test_lex_greater_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_lex_greater_unsat.cnf")
    a = satx.vector(size=2)
    b = satx.vector(size=2)
    assert a[0] == 0
    assert a[1] == 0
    assert b[0] == 0
    assert b[1] == 1
    gcc.lex_greater(a, b)
    assert not satx.satisfy(solver="slime")


def test_lex_greatereq_sat_equal():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_lex_greatereq_sat.cnf")
    a = satx.vector(size=2)
    b = satx.vector(size=2)
    assert a[0] == 0
    assert a[1] == 1
    assert b[0] == 0
    assert b[1] == 1
    gcc.lex_greatereq(a, b)
    assert satx.satisfy(solver="slime")
    assert tuple(v.value for v in a) >= tuple(v.value for v in b)


def test_lex_greatereq_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_lex_greatereq_unsat.cnf")
    a = satx.vector(size=2)
    b = satx.vector(size=2)
    assert a[0] == 0
    assert a[1] == 0
    assert b[0] == 0
    assert b[1] == 1
    gcc.lex_greatereq(a, b)
    assert not satx.satisfy(solver="slime")


def test_lex_chain_less_sat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_lex_chain_less_sat.cnf")
    v0 = satx.vector(size=2)
    v1 = satx.vector(size=2)
    v2 = satx.vector(size=2)
    assert v0[0] == 0
    assert v0[1] == 0
    assert v1[0] == 0
    assert v1[1] == 1
    assert v2[0] == 1
    assert v2[1] == 0
    gcc.lex_chain_less([v0, v1, v2])
    assert satx.satisfy(solver="slime")
    assert tuple(v.value for v in v0) < tuple(v.value for v in v1) < tuple(v.value for v in v2)


def test_lex_chain_less_unsat():
    satx.engine(bits=4, cnf_path="tests/tmp_gcc_lex_chain_less_unsat.cnf")
    v0 = satx.vector(size=2)
    v1 = satx.vector(size=2)
    v2 = satx.vector(size=2)
    assert v0[0] == 0
    assert v0[1] == 0
    assert v1[0] == 0
    assert v1[1] == 0
    assert v2[0] == 1
    assert v2[1] == 0
    gcc.lex_chain_less([v0, v1, v2])
    assert not satx.satisfy(solver="slime")


def test_inverse_sat():
    satx.engine(bits=3, cnf_path="tests/tmp_gcc_inverse_sat.cnf")
    fwd = satx.vector(size=3)
    inv = satx.vector(size=3)
    assert fwd[0] == 1
    assert fwd[1] == 2
    assert fwd[2] == 0
    assert inv[0] == 2
    assert inv[1] == 0
    assert inv[2] == 1
    gcc.inverse(fwd, inv)
    assert satx.satisfy(solver="slime")
    fv = [v.value for v in fwd]
    iv = [v.value for v in inv]
    assert all(iv[fv[i]] == i for i in range(3))
    assert all(fv[iv[i]] == i for i in range(3))


def test_inverse_unsat():
    satx.engine(bits=3, cnf_path="tests/tmp_gcc_inverse_unsat.cnf")
    fwd = satx.vector(size=3)
    inv = satx.vector(size=3)
    assert fwd[0] == 1
    assert fwd[1] == 2
    assert fwd[2] == 0
    assert inv[0] == 0
    assert inv[1] == 1
    assert inv[2] == 2
    gcc.inverse(fwd, inv)
    assert not satx.satisfy(solver="slime")


def test_circuit_sat():
    satx.engine(bits=3, cnf_path="tests/tmp_gcc_circuit_sat.cnf")
    succ = satx.vector(size=3)
    assert succ[0] == 1
    assert succ[1] == 2
    assert succ[2] == 0
    gcc.circuit(succ)
    assert satx.satisfy(solver="slime")
    s = [v.value for v in succ]
    seen = set()
    cur = 0
    for _ in range(3):
        seen.add(cur)
        cur = s[cur]
    assert seen == {0, 1, 2}
    assert cur == 0


def test_circuit_unsat():
    satx.engine(bits=3, cnf_path="tests/tmp_gcc_circuit_unsat.cnf")
    succ = satx.vector(size=3)
    assert succ[0] == 1
    assert succ[1] == 0
    assert succ[2] == 2
    gcc.circuit(succ)
    assert not satx.satisfy(solver="slime")


def test_bin_packing_sat():
    satx.engine(bits=6, cnf_path="tests/tmp_gcc_bin_packing_sat.cnf")
    bins = satx.vector(size=3)
    loads = satx.vector(size=2)
    sizes = [2, 1, 3]
    assert bins[0] == 0
    assert bins[1] == 1
    assert bins[2] == 0
    gcc.bin_packing(loads, bins, sizes)
    assert satx.satisfy(solver="slime")
    bv = [b.value for b in bins]
    lv = [l.value for l in loads]
    expect = [0, 0]
    for i, s in enumerate(sizes):
        expect[bv[i]] += s
    assert lv == expect


def test_bin_packing_unsat():
    satx.engine(bits=6, cnf_path="tests/tmp_gcc_bin_packing_unsat.cnf")
    bins = satx.vector(size=3)
    loads = satx.vector(size=2)
    sizes = [2, 1, 3]
    assert bins[0] == 0
    assert bins[1] == 1
    assert bins[2] == 0
    gcc.bin_packing(loads, bins, sizes)
    assert loads[0] == 4
    assert not satx.satisfy(solver="slime")


def test_bin_packing_capa_sat():
    satx.engine(bits=6, cnf_path="tests/tmp_gcc_bin_packing_capa_sat.cnf")
    bins = satx.vector(size=3)
    loads = satx.vector(size=2)
    sizes = [2, 1, 3]
    capa = [5, 2]
    assert bins[0] == 0
    assert bins[1] == 1
    assert bins[2] == 0
    gcc.bin_packing_capa(loads, bins, sizes, capa)
    assert satx.satisfy(solver="slime")
    assert loads[0].value <= capa[0]
    assert loads[1].value <= capa[1]


def test_bin_packing_capa_unsat():
    satx.engine(bits=6, cnf_path="tests/tmp_gcc_bin_packing_capa_unsat.cnf")
    bins = satx.vector(size=3)
    loads = satx.vector(size=2)
    sizes = [2, 1, 3]
    capa = [4, 2]
    assert bins[0] == 0
    assert bins[1] == 1
    assert bins[2] == 0
    gcc.bin_packing_capa(loads, bins, sizes, capa)
    assert not satx.satisfy(solver="slime")


def test_diffn_sat():
    satx.engine(bits=6, cnf_path="tests/tmp_gcc_diffn_sat.cnf")
    x = satx.vector(size=2)
    y = satx.vector(size=2)
    assert x[0] == 0
    assert y[0] == 0
    assert x[1] == 3
    assert y[1] == 0
    gcc.diffn(x, y, [2, 2], [2, 2])
    assert satx.satisfy(solver="slime")
    r0 = (x[0].value, y[0].value, 2, 2)
    r1 = (x[1].value, y[1].value, 2, 2)
    x0, y0, w0, h0 = r0
    x1, y1, w1, h1 = r1
    overlap = not (x0 + w0 <= x1 or x1 + w1 <= x0 or y0 + h0 <= y1 or y1 + h1 <= y0)
    assert not overlap


def test_diffn_unsat():
    satx.engine(bits=6, cnf_path="tests/tmp_gcc_diffn_unsat.cnf")
    x = satx.vector(size=2)
    y = satx.vector(size=2)
    assert x[0] == 0
    assert y[0] == 0
    assert x[1] == 1
    assert y[1] == 1
    gcc.diffn(x, y, [2, 2], [2, 2])
    assert not satx.satisfy(solver="slime")


def test_cumulative_sat():
    satx.engine(bits=6, cnf_path="tests/tmp_gcc_cumulative_sat.cnf")
    s0 = satx.integer()
    s1 = satx.integer()
    assert s0 == 0
    assert s1 == 1
    gcc.cumulative([s0, s1], duration=[2, 2], demand=[2, 2], limit=4, horizon=4)
    assert satx.satisfy(solver="slime")
    starts = [s0.value, s1.value]
    for t in range(4):
        use = 0
        for st, dur, dem in zip(starts, [2, 2], [2, 2]):
            if st <= t < st + dur:
                use += dem
        assert use <= 4


def test_cumulative_unsat():
    satx.engine(bits=6, cnf_path="tests/tmp_gcc_cumulative_unsat.cnf")
    s0 = satx.integer()
    s1 = satx.integer()
    assert s0 == 0
    assert s1 == 1
    gcc.cumulative([s0, s1], duration=[2, 2], demand=[2, 2], limit=3, horizon=4)
    assert not satx.satisfy(solver="slime")


def test_signed_minimum_sat():
    satx.engine(bits=5, signed=True, cnf_path="tests/tmp_gcc_signed_minimum_sat.cnf")
    xs = satx.vector(size=3)
    mn = satx.integer()
    assert xs[0] == -2
    assert xs[1] == 1
    assert xs[2] == 0
    gcc.minimum(mn, xs)
    assert satx.satisfy(solver="slime")
    assert mn.value == -2

