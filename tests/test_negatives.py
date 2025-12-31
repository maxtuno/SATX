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


def test_signed_sum_with_negatives():
    satx.engine(bits=4, signed=True, cnf_path="tests/tmp_neg_sum.cnf")
    x = satx.integer()
    y = satx.integer()
    assert x < 0
    assert y < 0
    assert x + y == -3
    assert satx.satisfy(solver="slime")
    assert x.value < 0
    assert y.value < 0
    assert x.value + y.value == -3


def test_signed_abs_difference_with_negatives():
    satx.engine(bits=5, signed=True, cnf_path="tests/tmp_neg_abs.cnf")
    x = satx.integer()
    y = satx.integer()
    assert x < 0
    assert y < 0
    assert abs(x - y) == 1
    assert satx.satisfy(solver="slime")
    assert x.value < 0
    assert y.value < 0
    assert abs(x.value - y.value) == 1


def test_signed_product_with_signs():
    satx.engine(bits=6, signed=True, cnf_path="tests/tmp_neg_mul.cnf")
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


def test_signed_mixed_comparisons():
    satx.engine(bits=6, signed=True, cnf_path="tests/tmp_neg_cmp.cnf")
    x = satx.integer()
    y = satx.integer()
    assert x < 0
    assert y > 0
    assert x < y
    assert x + y == 2
    assert satx.satisfy(solver="slime")
    assert x.value < 0
    assert y.value > 0
    assert x.value < y.value
    assert x.value + y.value == 2


def test_signed_negative_base_exponent():
    satx.engine(bits=8, signed=True, cnf_path="tests/tmp_neg_pow.cnf")
    base = satx.integer()
    exp = satx.integer()
    out = satx.integer()
    assert base == -2
    assert exp == 3
    assert base**exp == out
    assert out == -8
    assert satx.satisfy(solver="slime")
    assert base.value == -2
    assert exp.value == 3
    assert out.value == -8


def test_signed_overflow_in_addition_is_unsat():
    satx.engine(bits=4, signed=True, cnf_path="tests/tmp_neg_overflow_add.cnf")
    x = satx.integer()
    y = satx.integer()
    assert x == 7
    assert y == 1
    # 7 + 1 would overflow (wraps to -8 in 4-bit two's complement).
    assert x + y == -8
    assert not satx.satisfy(solver="slime")


def test_signed_overflow_in_multiplication_is_unsat():
    satx.engine(bits=4, signed=True, cnf_path="tests/tmp_neg_overflow_mul.cnf")
    x = satx.integer()
    y = satx.integer()
    assert x == 7
    assert y == 2
    # 7 * 2 would overflow (wraps to -2 in 4-bit two's complement).
    assert x * y == -2
    assert not satx.satisfy(solver="slime")
