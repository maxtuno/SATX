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


def test_case_inequaciones_simples():
    satx.engine(bits=10, cnf_path="tests/tmp_case_inequaciones_simples.cnf")
    x = satx.integer()
    y = satx.integer()
    assert 0 < x <= 3
    assert 0 < y <= 3
    assert x + y > 2
    assert satx.satisfy(solver="slime")
    assert 0 < x.value <= 3
    assert 0 < y.value <= 3
    assert x.value + y.value > 2


def test_case_circulo_pitagorico():
    satx.engine(bits=10, cnf_path="tests/tmp_case_circulo_pitagorico.cnf")
    x = satx.integer()
    y = satx.integer()
    assert x**2 + y**2 == 100
    assert satx.satisfy(solver="slime")
    assert x.value**2 + y.value**2 == 100


def test_case_abs_diferencia():
    satx.engine(4, cnf_path="tests/tmp_case_abs_diferencia.cnf")
    x = satx.integer()
    y = satx.integer()
    assert abs(x - y) == 1
    assert x != satx.oo()
    assert y != satx.oo()
    assert satx.satisfy(solver="slime")
    assert abs(x.value - y.value) == 1
    assert x.value != satx.oo()
    assert y.value != satx.oo()


def test_case_negativos_suma():
    satx.engine(bits=4, signed=True, cnf_path="tests/tmp_case_negativos_suma.cnf")
    x = satx.integer()
    y = satx.integer()
    assert x < 0
    assert y < 0
    assert x + y == -3
    assert satx.satisfy(solver="slime")
    assert x.value < 0
    assert y.value < 0
    assert x.value + y.value == -3


def test_case_negativos_producto_signo():
    satx.engine(bits=5, signed=True, cnf_path="tests/tmp_case_negativos_producto_signo.cnf")
    a = satx.integer()
    b = satx.integer()
    c = satx.integer()
    assert 0 < a <= 3
    assert 0 < b <= 3
    assert 0 < c <= 9
    assert (-a) * b == -c
    assert satx.satisfy(solver="slime")
    assert 0 < a.value <= 3
    assert 0 < b.value <= 3
    assert 0 < c.value <= 9
    assert (-a.value) * b.value == -c.value


def test_case_subset_sum():
    universe = [3, 5, 7, 9]
    target = 12
    satx.engine(target.bit_length(), cnf_path="tests/tmp_case_subset_sum.cnf")
    idx, subset, _ = satx.subsets(universe, complement=True)
    assert sum(subset) == target
    assert satx.satisfy(solver="slime")
    chosen = [universe[i] for i in range(len(universe)) if not idx.binary[i]]
    subset_values = [x.value for x in subset]
    assert sum(chosen) == target
    assert sum(subset_values) == target
    assert all(v in (0, universe[i]) for i, v in enumerate(subset_values))


def test_case_peqnp_exponente_cuadratico():
    satx.engine(32, cnf_path="tests/tmp_case_peqnp_exponente_cuadratico.cnf")
    _2 = satx.constant(2)
    n = satx.integer()
    x = satx.integer()
    assert _2**n - 7 == x**2
    assert satx.satisfy(solver="slime")
    assert pow(2, n.value) - 7 == x.value**2


def test_case_peqnp_cubica_vs_potencia():
    satx.engine(16, cnf_path="tests/tmp_case_peqnp_cubica_vs_potencia.cnf")
    _2 = satx.constant(2)
    n = satx.integer()
    k = satx.integer()
    assert n**3 + 10 == _2**k + 5 * n
    assert satx.satisfy(solver="slime")
    assert n.value**3 + 10 == pow(2, k.value) + 5 * n.value


def test_case_peqnp_diophantina():
    satx.engine(16, cnf_path="tests/tmp_case_peqnp_diophantina.cnf")
    a = satx.integer()
    b = satx.integer()
    assert a**2 == b**3 + 1
    assert satx.satisfy(solver="slime")
    assert a.value**2 == b.value**3 + 1


def test_case_peqnp_exponencial():
    satx.engine(32, cnf_path="tests/tmp_case_peqnp_exponencial.cnf")
    _2 = satx.constant(2)
    _3 = satx.constant(3)
    x = satx.integer()
    y = satx.integer()
    assert _3**x == y * _2**x + 1
    assert satx.satisfy(solver="slime")
    assert pow(3, x.value) == y.value * pow(2, x.value) + 1


def test_case_peqnp_factorizacion_rsa():
    rsa = 3007
    satx.engine(rsa.bit_length(), cnf_path="tests/tmp_case_peqnp_factorizacion_rsa.cnf")
    p = satx.integer()
    q = satx.integer()
    assert p * q == rsa
    assert satx.satisfy(solver="slime")
    assert p.value * q.value == rsa


def test_case_peqnp_diferencia_de_cuadrados():
    rsa = 3007
    satx.engine(rsa.bit_length() + 1, cnf_path="tests/tmp_case_peqnp_diferencia_de_cuadrados.cnf")
    p = satx.integer()
    q = satx.integer()
    assert p**2 - q**2 == rsa
    assert q < p
    assert satx.satisfy(solver="slime")
    assert p.value**2 - q.value**2 == rsa
    assert q.value < p.value
