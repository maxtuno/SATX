from fractions import Fraction
import warnings

import pytest
import satx
from satx.unit import Unit


def test_fixed_add():
    satx.engine(bits=16, cnf_path="tests/tmp_fixed_add.cnf")
    x = satx.fixed(scale=100)
    y = satx.fixed(scale=100)
    assert x == satx.fixed_const(1.20, scale=100)
    assert y == satx.fixed_const(0.30, scale=100)
    assert x + y == satx.fixed_const(1.50, scale=100)
    assert satx.satisfy(solver="slime")
    assert x.value == Fraction(6, 5)
    assert y.value == Fraction(3, 10)


def test_fixed_mul_exact():
    satx.engine(bits=16, cnf_path="tests/tmp_fixed_mul_exact.cnf")
    a = satx.fixed_const(1.25, scale=100)
    b = satx.fixed_const(2.00, scale=100)
    c = satx.fixed_const(2.50, scale=100)
    prod = a * b
    assert prod == c
    assert satx.satisfy(solver="slime")
    assert prod.value == Fraction(5, 2)
    assert a.value * b.value == c.value


def test_scale_mismatch_raises():
    satx.engine(bits=16, cnf_path="tests/tmp_fixed_scale_mismatch.cnf")
    x = satx.fixed(scale=100)
    y = satx.fixed(scale=1000)
    with pytest.raises(ValueError, match="scale mismatch"):
        _ = x + y


def test_integer_scale_returns_fixed():
    satx.engine(bits=8, cnf_path="tests/tmp_fixed_integer_scale.cnf")
    x = satx.integer(scale=100)
    assert isinstance(x, satx.Fixed)
    assert isinstance(x.raw, Unit)


def test_integer_scale_pow_example():
    satx.engine(bits=16, cnf_path="tests/tmp_fixed_pow_example.cnf")
    x = satx.integer(scale=100)
    y = satx.integer(scale=100)
    total = satx.integer(scale=100)
    assert x == satx.fixed_const(1.00, scale=100)
    assert y == satx.fixed_const(1.00, scale=100)
    assert x**3 + y**3 == total
    assert satx.satisfy(solver="slime")
    assert total.value == Fraction(2, 1)


def test_fixed_pow_rejects_float():
    satx.engine(bits=8, cnf_path="tests/tmp_fixed_pow_type.cnf")
    x = satx.integer(scale=100)
    with pytest.raises(TypeError, match="Fixed \\*\\* exp requiere exp Python int"):
        _ = x**2.0


def test_fixed_eq_int_promotes():
    satx.engine(bits=16, cnf_path="tests/tmp_fixed_eq_int_promotes.cnf")
    x = satx.integer(scale=100)
    y = satx.integer(scale=100)
    assert x == satx.fixed_const(1.00, scale=100)
    assert y == satx.fixed_const(0, scale=100)
    assert x**2 + y**3 == 1
    assert satx.satisfy(solver="slime")


def test_fixed_rmul_int():
    satx.engine(bits=16, cnf_path="tests/tmp_fixed_rmul_int.cnf")
    x = satx.integer(scale=100)
    assert x == satx.fixed_const(1.00, scale=100)
    assert 2 * x == x + x
    assert satx.satisfy(solver="slime")


def test_fixed_print_clean():
    satx.engine(bits=16, cnf_path="tests/tmp_fixed_print_clean.cnf")
    x = satx.integer(scale=100)
    assert x == satx.fixed_const(1.25, scale=100)
    assert satx.satisfy(solver="slime")
    rendered = str(x)
    assert "Fixed(" not in rendered
    assert rendered == "1.25"


def test_fixed_repr_is_numeric():
    satx.engine(bits=8, cnf_path="tests/tmp_fixed_repr.cnf")
    x = satx.fixed_const(Fraction(1, 3), scale=3)
    assert satx.satisfy(solver="slime")
    assert "Fixed(" not in repr(x)
    assert repr(x) == "1/3"
    assert "Fixed(" in x.debug_repr()


def test_clear_accepts_fixed():
    satx.engine(bits=8, cnf_path="tests/tmp_fixed_clear.cnf")
    x = satx.integer(scale=10)
    y = satx.integer(scale=10)
    assert 0 <= x <= 1
    assert 0 <= y <= 1
    assert x + y == 1
    assert satx.satisfy(solver="slime")
    first = (x.value, y.value)
    satx.clear([x, y])
    assert x.value is None
    assert y.value is None
    assert satx.satisfy(solver="slime")
    second = (x.value, y.value)
    assert first != second


def test_fixed_advice_smoke():
    satx.engine(bits=10, signed=True, cnf_path="tests/tmp_fixed_advice.cnf")
    x = satx.integer(scale=8)
    advice = satx.fixed_advice(x, degree=2)
    assert advice["bits"] == 10
    assert advice["signed"] is True
    scales = advice["scales"]
    assert scales[0]["scale"] == 8
    assert scales[0]["raw_max_abs"] == 511


def test_fixed_advice_respects_engine_defaults():
    satx.engine(bits=20, fixed_default=True, fixed_scale=3, signed=True, cnf_path="tests/tmp_fixed_advice_defaults.cnf")
    advice = satx.fixed_advice()
    assert advice["bits"] == 20
    assert advice["signed"] is True
    assert advice["scales"][0]["scale"] == 3


def test_fixed_promotions_with_ints():
    satx.engine(bits=12, cnf_path="tests/tmp_fixed_promotions.cnf")
    x = satx.integer(scale=10)
    assert x == satx.fixed_const(1.5, scale=10)
    assert 2 + x == satx.fixed_const(3.5, scale=10)
    assert x - 1 == satx.fixed_const(0.5, scale=10)
    assert x < 2
    assert satx.satisfy(solver="slime")


def test_float_times_fixed():
    satx.engine(bits=12, fixed_default=True, fixed_scale=10, cnf_path="tests/tmp_fixed_float_mul.cnf")
    x = satx.integer()
    assert x == satx.fixed_const(2.0, scale=10)
    prod = 2.5 * x
    assert prod == satx.fixed_const(5.0, scale=10)
    assert satx.satisfy(solver="slime")


def test_float_times_unit_default_fixed():
    satx.engine(bits=12, fixed_default=True, fixed_scale=10, cnf_path="tests/tmp_fixed_float_mul_unit.cnf")
    s = satx.integer(force_int=True)
    cost = 3.2 * s
    assert isinstance(cost, satx.Fixed)
    assert s == 2
    assert cost == satx.fixed_const(6.4, scale=10)
    assert satx.satisfy(solver="slime")


def test_unit_le_fixed():
    satx.engine(bits=12, fixed_default=True, fixed_scale=10, cnf_path="tests/tmp_fixed_unit_le.cnf")
    y = satx.integer()
    s = satx.integer(force_int=True)
    assert y == satx.fixed_const(1.0, scale=10)
    assert s == 1
    assert s <= 3 * y
    assert satx.satisfy(solver="slime")


def test_fixed_eq_unit():
    satx.engine(bits=8, cnf_path="tests/tmp_fixed_eq_unit.cnf")
    u = satx.integer()
    x = satx.integer(scale=10)
    assert u == 3
    assert x == u
    assert satx.satisfy(solver="slime")
    assert u.value == 3
    assert x.value == Fraction(3, 1)


def test_fixed_signed_negative():
    satx.engine(bits=8, signed=True, cnf_path="tests/tmp_fixed_signed_neg.cnf")
    x = satx.fixed_const(-1.2, scale=10)
    assert x < 0
    assert satx.satisfy(solver="slime")
    assert x.value == Fraction(-6, 5)


def test_fixed_unsigned_negative_rejected():
    satx.engine(bits=8, signed=False, cnf_path="tests/tmp_fixed_unsigned_neg.cnf")
    with pytest.raises(ValueError, match="unsigned"):
        _ = satx.fixed_const(-1, scale=10)


def test_fixed_mul_overflow_unsat():
    satx.engine(bits=8, cnf_path="tests/tmp_fixed_mul_overflow.cnf")
    a = satx.fixed_const(1, scale=50)
    b = satx.fixed_const(1, scale=50)
    c = satx.fixed_const(1, scale=50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert a * b == c
    assert not satx.satisfy(solver="slime")


def test_fixed_pow_small():
    satx.engine(bits=16, cnf_path="tests/tmp_fixed_pow_small.cnf")
    x = satx.integer(scale=10)
    assert x == satx.fixed_const(2, scale=10)
    assert x**0 == satx.fixed_const(1, scale=10)
    assert x**1 == x
    assert x**2 == satx.fixed_const(4, scale=10)
    assert x**3 == satx.fixed_const(8, scale=10)
    assert satx.satisfy(solver="slime")


def test_engine_fixed_default_integer_vector_matrix():
    satx.engine(bits=8, fixed_default=True, fixed_scale=100, cnf_path="tests/tmp_fixed_default.cnf")
    x = satx.integer()
    assert isinstance(x, satx.Fixed)
    assert x.scale == 100
    v = satx.vector(size=3)
    assert all(isinstance(item, satx.Fixed) for item in v)
    m = satx.matrix(dimensions=(2, 2))
    assert isinstance(m[0][0], satx.Fixed)


def test_fixed_default_override_to_unit():
    satx.engine(bits=12, fixed_default=True, fixed_scale=10, cnf_path="tests/tmp_fixed_default_override.cnf")
    u = satx.integer(force_int=True)
    assert isinstance(u, Unit)
    v = satx.vector(size=2, fixed=False)
    assert all(isinstance(item, Unit) for item in v)
    m = satx.matrix(dimensions=(1, 2), fixed=True, scale=1000)
    assert isinstance(m[0][0], satx.Fixed)
    assert m[0][0].scale == 1000


def test_fixed_pow_limit():
    satx.engine(bits=8, cnf_path="tests/tmp_fixed_pow_limit.cnf", max_fixed_pow=2)
    x = satx.integer(scale=10)
    with pytest.raises(ValueError, match="max_fixed_pow"):
        _ = x**3


def test_as_int_policies():
    satx.engine(bits=8, signed=True, cnf_path="tests/tmp_fixed_as_int.cnf")
    x = satx.fixed_const(1.5, scale=10)
    floor = satx.as_int(x, policy="floor")
    ceil = satx.as_int(x, policy="ceil")
    round_ = satx.as_int(x, policy="round")
    y = satx.integer(scale=10)
    q = satx.as_int(y, policy="floor")
    assert isinstance(floor, Unit)
    assert isinstance(ceil, Unit)
    assert isinstance(round_, Unit)
    assert isinstance(q, Unit)
    assert satx.satisfy(solver="slime")
    assert floor.value == 1
    assert ceil.value == 2
    assert round_.value == 2
    with pytest.raises(ValueError, match="exact"):
        satx.as_int(x, policy="exact")


def test_fixed_mul_floor_round_constants():
    satx.engine(bits=8, cnf_path="tests/tmp_fixed_mul_round.cnf")
    a = satx.as_fixed(satx.constant(15), scale=10)
    b = satx.as_fixed(satx.constant(13), scale=10)
    floor = satx.fixed_mul_floor(a, b)
    round_ = satx.fixed_mul_round(a, b)
    assert floor == satx.fixed_const(1.9, scale=10)
    assert round_ == satx.fixed_const(2.0, scale=10)


def test_fixed_compare_fraction_after_clear():
    satx.engine(bits=8, cnf_path="tests/tmp_fixed_fraction_cmp.cnf")
    x = satx.integer(scale=1)
    assert x == satx.fixed_const(3, scale=1)
    assert satx.satisfy(solver="slime")
    optimal = x.value
    satx.clear([x])
    assert x < optimal
    assert not satx.satisfy(solver="slime")


def test_fixed_fuzz_smoke():
    import random

    rng = random.Random(1337)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for idx in range(200):
            bits = rng.randint(8, 12)
            signed = rng.choice([False, True])
            raw_max_abs = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
            scale = rng.randint(2, min(1000, raw_max_abs))
            satx.engine(bits=bits, signed=signed, cnf_path=f"tests/tmp_fixed_fuzz_{idx}.cnf")
            x = satx.integer(scale=scale)
            y = satx.integer(scale=scale)
            z = satx.integer(scale=scale)
            op = rng.choice(["add", "mul", "pow"])
            if op == "add":
                assert x + y == z
            elif op == "mul":
                assert x * y == z
            else:
                exp = rng.randint(0, 3)
                assert x**exp == z
            if rng.random() < 0.5:
                assert x >= 0
            if rng.random() < 0.5:
                assert y == 1
            try:
                satx.satisfy(solver="slime")
            except Exception as exc:
                raise AssertionError(
                    f"fuzz crash idx={idx} bits={bits} signed={signed} scale={scale} op={op}"
                ) from exc
