"""
Fixed-point decimals for SATX.

This module introduces a small wrapper type (`Fixed`) that represents a decimal
value `raw / scale`, where `raw` is a SATX `Unit` (fixed-width bit-vector
integer) and `scale` is a positive Python integer (typically `10**k`).

Design goals:
- No floating-point gates: everything is encoded via existing `Unit` arithmetic.
- Preserve SATX overflow semantics: any overflow in underlying `Unit` ops stays UNSAT.
- Keep rounding explicit: `Fixed.__mul__` is exact via a rescale constraint; no hidden rounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
from typing import Any, Optional, Union
import warnings

from .rational import Rational
from .unit import Unit
from . import stdlib as _stdlib


NumberLike = Union[int, float, Decimal, Fraction]
_WARNED_OVERFLOW = set()


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _check_scale_fits_unit(scale: int, raw: Unit) -> None:
    bits = raw.bits
    if bits is None:
        return
    if raw.alu.signed:
        max_pos = (1 << (bits - 1)) - 1
        if not (1 <= scale <= max_pos):
            raise ValueError(
                f"scale={scale} is not representable as a positive {bits}-bit signed SATX integer "
                f"(max {max_pos}); increase engine/variable bit-width or use a smaller scale"
            )
    else:
        max_pos = (1 << bits) - 1
        if not (1 <= scale <= max_pos):
            raise ValueError(
                f"scale={scale} is not representable as a positive {bits}-bit unsigned SATX integer "
                f"(max {max_pos}); increase engine/variable bit-width or use a smaller scale"
            )


def _check_raw_value_fits_engine(raw_value: int) -> None:
    _stdlib.check_engine()
    bits = _stdlib.csp.bits
    if _stdlib.csp.signed:
        min_v = -(1 << (bits - 1))
        max_v = (1 << (bits - 1)) - 1
        if not (min_v <= raw_value <= max_v):
            raise ValueError(
                f"raw value {raw_value} does not fit in current signed engine width ({bits} bits, "
                f"range [{min_v}, {max_v}]); increase `satx.engine(bits=...)`"
            )
    else:
        if raw_value < 0:
            raise ValueError(
                f"raw value {raw_value} is negative, but the current engine is unsigned; "
                f"use `satx.engine(..., signed=True)` for negative fixed-point values"
            )
        max_v = (1 << bits) - 1
        if raw_value > max_v:
            raise ValueError(
                f"raw value {raw_value} does not fit in current unsigned engine width ({bits} bits, "
                f"max {max_v}); increase `satx.engine(bits=...)`"
            )


def _encode_fixed_const(x: NumberLike, scale: int) -> int:
    if isinstance(x, bool):
        raise TypeError("fixed_const does not accept bool (use int)")
    if isinstance(x, int):
        return x * scale
    if isinstance(x, Fraction):
        scaled_num = x.numerator * scale
        if scaled_num % x.denominator != 0:
            raise ValueError(
                f"Fraction {x} is not exactly representable at scale={scale}; "
                f"use a larger scale or pass a float/Decimal to allow rounding explicitly"
            )
        return scaled_num // x.denominator
    if isinstance(x, Decimal):
        scaled = x * Decimal(scale)
        return int(scaled.to_integral_value())
    if isinstance(x, float):
        return int(round(x * scale))
    raise TypeError(f"Unsupported type for fixed_const: {type(x).__name__}")


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal, Fraction)) and not isinstance(value, bool)


def _coerce_numeric_value(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction.from_float(value)
    raise TypeError(f"Unsupported numeric value: {type(value).__name__}")


def _raw_max_abs(bits: int, signed: bool) -> int:
    if bits <= 0:
        return 0
    return (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1


def _maybe_warn_overflow(scale: int, bits: int, signed: bool) -> None:
    key = (bits, signed, scale)
    if key in _WARNED_OVERFLOW:
        return
    raw_max_abs = _raw_max_abs(bits, signed)
    if raw_max_abs and scale * scale > raw_max_abs:
        warnings.warn(
            "Fixed scale may overflow with current bit-width; increase bits or reduce scale",
            UserWarning,
            stacklevel=3,
        )
        _WARNED_OVERFLOW.add(key)


@dataclass(frozen=True)
class Fixed:
    raw: Unit
    scale: int

    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.raw, Unit):
            raise TypeError(f"raw must be a satx.Unit, got {type(self.raw).__name__}")
        scale = _require_positive_int("scale", self.scale)
        object.__setattr__(self, "scale", scale)
        _check_scale_fits_unit(scale, self.raw)

    @property
    def value(self) -> Optional[Fraction]:
        if self.raw.value is None:
            return None
        return Fraction(self.raw.value, self.scale)

    def __repr__(self) -> str:
        return str(self)

    def debug_repr(self) -> str:
        if self.raw.value is None:
            return f"Fixed(raw={self.raw}, scale={self.scale})"
        return f"Fixed(value={self.value}, scale={self.scale})"

    def _require_compatible(self, other: Any, op: str) -> "Fixed":
        if isinstance(other, Fixed):
            if self.scale != other.scale:
                raise ValueError(f"scale mismatch for {op}: {self.scale} != {other.scale}")
            if self.raw.alu is not other.raw.alu:
                raise ValueError(f"cannot {op} Fixed values from different SATX engines")
            return other
        if _is_numeric_value(other):
            other_fixed = fixed_const(other, scale=self.scale)
            if self.raw.alu is not other_fixed.raw.alu:
                raise ValueError(f"cannot {op} Fixed values from different SATX engines")
            return other_fixed
        if isinstance(other, Unit):
            if self.raw.alu is not other.alu:
                raise ValueError(f"cannot {op} Fixed values from different SATX engines")
            return Fixed(other * self.scale, self.scale)
        raise TypeError(f"unsupported operand type(s) for {op}: 'Fixed' and '{type(other).__name__}'")

    def __neg__(self):
        if self.raw.value is not None:
            return -self.value
        return Fixed(-self.raw, self.scale)

    def __add__(self, other: Any):
        if self.raw.value is not None and isinstance(other, int) and not isinstance(other, bool):
            return self.value + other
        other = self._require_compatible(other, "+")
        if self.raw.value is not None:
            return self.value + other.value
        return Fixed(self.raw + other.raw, self.scale)

    def __radd__(self, other: Any):
        if isinstance(other, int) and not isinstance(other, bool) and other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other: Any):
        if self.raw.value is not None and isinstance(other, int) and not isinstance(other, bool):
            return self.value - other
        other = self._require_compatible(other, "-")
        if self.raw.value is not None:
            return self.value - other.value
        return Fixed(self.raw - other.raw, self.scale)

    def __rsub__(self, other: Any):
        if self.raw.value is not None and isinstance(other, int) and not isinstance(other, bool):
            return other - self.value
        other = self._require_compatible(other, "-")
        if self.raw.value is not None:
            return other.value - self.value
        return Fixed(other.raw - self.raw, self.scale)

    def __mul__(self, other: Any):
        if self.raw.value is not None and isinstance(other, int) and not isinstance(other, bool):
            return self.value * other
        other = self._require_compatible(other, "*")
        if self.raw.value is not None:
            return self.value * other.value
        _maybe_warn_overflow(self.scale, self.raw.bits, self.raw.alu.signed)

        r = Unit(self.raw.alu, bits=self.raw.bits)
        self.raw.alu.variables.append(r)
        assert (self.raw * other.raw) == (r * self.scale)
        return Fixed(r, self.scale)

    def __rmul__(self, other: Any):
        return self.__mul__(other)

    def __pow__(self, power: Any, modulo: Optional[int] = None):
        if modulo is not None:
            raise TypeError("Fixed does not support modular exponentiation")
        if isinstance(power, bool) or not isinstance(power, int):
            raise TypeError(
                "Fixed ** exp requiere exp Python int; para exponente variable usa satx.integer() sin scale (Unit)"
            )
        if power < 0:
            raise ValueError("power must be >= 0")
        limit = getattr(self.raw.alu, "max_fixed_pow", None)
        if limit is not None and power > limit:
            raise ValueError(
                f"Fixed exponent {power} exceeds max_fixed_pow={limit}; "
                f"use satx.engine(..., max_fixed_pow=...) or a Unit exponent"
            )
        if self.raw.value is not None:
            return self.value ** power
        if power == 0:
            return fixed_const(1, scale=self.scale)
        if power == 1:
            return self
        _maybe_warn_overflow(self.scale, self.raw.bits, self.raw.alu.signed)

        result = fixed_const(1, scale=self.scale)
        base = self
        exp = power
        while exp > 0:
            if exp & 1:
                result = result * base
            exp >>= 1
            if exp:
                base = base * base
        return result

    def __truediv__(self, other: Any):
        raise TypeError("Fixed division is not implicit; use satx.fixed_div_exact(a, b)")

    def __rtruediv__(self, other: Any):
        raise TypeError("Fixed division is not implicit; use satx.fixed_div_exact(a, b)")

    def __eq__(self, other: Any):  # type: ignore[override]
        if self.raw.value is not None:
            if _is_numeric_value(other):
                return self.value == _coerce_numeric_value(other)
        other = self._require_compatible(other, "==")
        if self.raw.value is not None and other.raw.value is not None:
            return self.raw.value == other.raw.value
        assert self.raw == other.raw
        return self

    def __ne__(self, other: Any):  # type: ignore[override]
        if self.raw.value is not None and _is_numeric_value(other):
            return self.value != _coerce_numeric_value(other)
        other = self._require_compatible(other, "!=")
        if self.raw.value is not None and other.raw.value is not None:
            return self.raw.value != other.raw.value
        return self.raw != other.raw

    def __lt__(self, other: Any):
        if self.raw.value is not None:
            if _is_numeric_value(other):
                return self.value < _coerce_numeric_value(other)
        other = self._require_compatible(other, "<")
        if self.raw.value is not None and other.raw.value is not None:
            return self.raw.value < other.raw.value
        assert self.raw < other.raw
        return self

    def __le__(self, other: Any):
        if self.raw.value is not None:
            if _is_numeric_value(other):
                return self.value <= _coerce_numeric_value(other)
        other = self._require_compatible(other, "<=")
        if self.raw.value is not None and other.raw.value is not None:
            return self.raw.value <= other.raw.value
        assert self.raw <= other.raw
        return self

    def __gt__(self, other: Any):
        if self.raw.value is not None:
            if _is_numeric_value(other):
                return self.value > _coerce_numeric_value(other)
        other = self._require_compatible(other, ">")
        if self.raw.value is not None and other.raw.value is not None:
            return self.raw.value > other.raw.value
        assert self.raw > other.raw
        return self

    def __ge__(self, other: Any):
        if self.raw.value is not None:
            if _is_numeric_value(other):
                return self.value >= _coerce_numeric_value(other)
        other = self._require_compatible(other, ">=")
        if self.raw.value is not None and other.raw.value is not None:
            return self.raw.value >= other.raw.value
        assert self.raw >= other.raw
        return self

    def __str__(self) -> str:
        if self.raw.value is None:
            return "<?>"
        scale = self.scale
        power = scale
        decimals = 0
        while power % 10 == 0:
            power //= 10
            decimals += 1
        if power == 1:
            value = self.raw.value
            if decimals == 0:
                return str(value)
            sign = "-" if value < 0 else ""
            abs_val = abs(value)
            whole = abs_val // scale
            frac = abs_val % scale
            return f"{sign}{whole}.{str(frac).rjust(decimals, '0')}"
        frac = Fraction(self.raw.value, scale)
        return f"{frac.numerator}/{frac.denominator}"


def fixed(*, scale: int = 100, bits: Optional[int] = None) -> Fixed:
    """
    Create a fresh fixed-point variable with underlying raw integer `Unit`.

    `value = raw / scale`
    """
    scale = _require_positive_int("scale", scale)
    raw = _stdlib.integer(bits=bits)
    return Fixed(raw, scale)


def fixed_const(x: NumberLike, *, scale: int = 100) -> Fixed:
    """
    Create a fixed-point constant by encoding `raw = round(x * scale)`.

    - int: exact (`raw = x * scale`)
    - float: rounded via Python `round`
    - Decimal: rounded via `to_integral_value` (context rounding)
    - Fraction: exact only if divisible by denominator, otherwise raises ValueError
    """
    scale = _require_positive_int("scale", scale)
    raw_value = _encode_fixed_const(x, scale)
    _check_raw_value_fits_engine(raw_value)
    raw = _stdlib.constant(raw_value)
    return Fixed(raw, scale)


def as_fixed(u: Unit, scale: int) -> Fixed:
    scale = _require_positive_int("scale", scale)
    return Fixed(u, scale)


def to_rational(f: Fixed) -> Rational:
    if not isinstance(f, Fixed):
        raise TypeError(f"to_rational expects Fixed, got {type(f).__name__}")
    denom = Unit(f.raw.alu, value=f.scale)
    return Rational(f.raw, denom)


def fixed_div_exact(a: Fixed, b: Fixed, *, scale: Optional[int] = None) -> Union[Rational, Fixed]:
    """
    Exact fixed-point division helper.

    If `scale is None` (default):
        returns `Rational(a.raw, b.raw)` (scale cancels when a.scale == b.scale).

    If `scale` is provided:
        returns a `Fixed(r, scale)` and enforces exact divisibility via:
            a.raw * scale == r * b.raw
    """
    if not isinstance(a, Fixed) or not isinstance(b, Fixed):
        raise TypeError("fixed_div_exact expects (Fixed, Fixed)")
    if a.scale != b.scale:
        raise ValueError(f"scale mismatch for division: {a.scale} != {b.scale}")
    if a.raw.alu is not b.raw.alu:
        raise ValueError("cannot divide Fixed values from different SATX engines")

    if scale is None:
        return Rational(a.raw, b.raw)

    scale = _require_positive_int("scale", scale)
    _check_scale_fits_unit(scale, a.raw)
    r = Unit(a.raw.alu, bits=a.raw.bits)
    a.raw.alu.variables.append(r)
    assert (a.raw * scale) == (r * b.raw)
    return Fixed(r, scale)


def _as_int_from_value(raw_value: int, scale: int, policy: str) -> int:
    if policy == "exact":
        if raw_value % scale != 0:
            raise ValueError("fixed value is not exactly representable as an integer at this scale")
        return raw_value // scale
    if policy == "floor":
        return raw_value // scale
    if policy == "ceil":
        return -((-raw_value) // scale)
    if policy == "round":
        sign = -1 if raw_value < 0 else 1
        abs_val = abs(raw_value)
        q = abs_val // scale
        r = abs_val % scale
        half_up = (scale + 1) // 2
        if r >= half_up:
            q += 1
        return sign * q
    raise ValueError(f"unknown policy: {policy}")


def as_int(f: Fixed, policy: str = "exact") -> Union[Unit, int]:
    """
    Convert a Fixed value to an integer Unit using an explicit rounding policy.

    Policies:
    - exact: require raw % scale == 0 (adds a constraint), then return raw // scale
    - floor: floor(raw / scale)
    - ceil: ceil(raw / scale)
    - round: round-half-up (away from zero)

    If the Fixed is already solved, returns a Python int.
    """
    if not isinstance(f, Fixed):
        raise TypeError(f"as_int expects Fixed, got {type(f).__name__}")
    if not isinstance(policy, str):
        raise TypeError(f"policy must be a str, got {type(policy).__name__}")
    if policy not in {"exact", "floor", "ceil", "round"}:
        raise ValueError(f"policy must be one of: exact, floor, ceil, round")
    if f.raw.value is not None:
        return _as_int_from_value(f.raw.value, f.scale, policy)

    raw = f.raw
    scale = f.scale
    if policy == "exact":
        assert raw % scale == 0
        return raw // scale
    if policy == "floor":
        return raw // scale

    q = raw // scale
    r = raw % scale
    alu = raw.alu
    if policy == "ceil":
        nonzero = alu.or_gate(r.block)
        indicator = alu.one.iff(nonzero, alu.zero)
        return q + indicator
    half_up = (scale + 1) // 2
    cond = alu.bv_ule_gate(alu.create_constant(half_up), r.block)
    indicator = alu.one.iff(cond, alu.zero)
    return q + indicator


def fixed_mul_floor(a: Fixed, b: Fixed) -> Fixed:
    """
    Multiply two Fixed values and floor the rescale to the same scale.
    """
    if not isinstance(a, Fixed) or not isinstance(b, Fixed):
        raise TypeError("fixed_mul_floor expects (Fixed, Fixed)")
    if a.scale != b.scale:
        raise ValueError(f"scale mismatch for multiplication: {a.scale} != {b.scale}")
    if a.raw.alu is not b.raw.alu:
        raise ValueError("cannot multiply Fixed values from different SATX engines")
    prod = a.raw * b.raw
    if isinstance(prod, int):
        raw_value = _as_int_from_value(prod, a.scale, "floor")
        _check_raw_value_fits_engine(raw_value)
        return Fixed(_stdlib.constant(raw_value), a.scale)
    raw = as_int(Fixed(prod, a.scale), policy="floor")
    if isinstance(raw, Unit):
        return Fixed(raw, a.scale)
    _check_raw_value_fits_engine(raw)
    return Fixed(_stdlib.constant(raw), a.scale)


def fixed_mul_round(a: Fixed, b: Fixed) -> Fixed:
    """
    Multiply two Fixed values and round the rescale (round-half-up) to the same scale.
    """
    if not isinstance(a, Fixed) or not isinstance(b, Fixed):
        raise TypeError("fixed_mul_round expects (Fixed, Fixed)")
    if a.scale != b.scale:
        raise ValueError(f"scale mismatch for multiplication: {a.scale} != {b.scale}")
    if a.raw.alu is not b.raw.alu:
        raise ValueError("cannot multiply Fixed values from different SATX engines")
    prod = a.raw * b.raw
    if isinstance(prod, int):
        raw_value = _as_int_from_value(prod, a.scale, "round")
        _check_raw_value_fits_engine(raw_value)
        return Fixed(_stdlib.constant(raw_value), a.scale)
    raw = as_int(Fixed(prod, a.scale), policy="round")
    if isinstance(raw, Unit):
        return Fixed(raw, a.scale)
    _check_raw_value_fits_engine(raw)
    return Fixed(_stdlib.constant(raw), a.scale)


def fixed_advice(*vars: Any, degree: int = 2, expected_max: Optional[float] = None) -> dict:
    _stdlib.check_engine()
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise TypeError(f"degree must be a Python int, got {type(degree).__name__}")
    if degree <= 0:
        raise ValueError("degree must be > 0")
    if expected_max is not None:
        if isinstance(expected_max, bool) or not isinstance(expected_max, (int, float)):
            raise TypeError("expected_max must be a number")
        if expected_max <= 0:
            raise ValueError("expected_max must be > 0")

    bits = _stdlib.csp.bits
    signed = _stdlib.csp.signed
    scales = set()
    if vars:
        for var in vars:
            if isinstance(var, Fixed):
                scales.add(var.scale)
            elif isinstance(var, Unit):
                scales.add(1)
            else:
                raise TypeError(f"fixed_advice expects Fixed or Unit, got {type(var).__name__}")
    else:
        if getattr(_stdlib.csp, "default_is_fixed", False):
            scales.add(getattr(_stdlib.csp, "default_scale", 1))
        else:
            scales.add(1)

    raw_max_abs = _raw_max_abs(bits, signed)
    entries = []
    for scale in sorted(scales):
        resolution = 1.0 / scale
        value_max_abs = raw_max_abs / scale if scale else 0.0
        safe_mul_value_max = math.sqrt(raw_max_abs) / scale if scale else 0.0
        safe_pow_value_max = (raw_max_abs / (scale ** 2)) ** (1.0 / degree) if scale else 0.0
        entry = {
            "scale": scale,
            "resolution": resolution,
            "raw_max_abs": raw_max_abs,
            "value_max_abs": value_max_abs,
            "safe_mul_value_max": safe_mul_value_max,
            "safe_pow_value_max": safe_pow_value_max,
        }
        if expected_max is not None:
            bits_suggested = math.ceil(1 + math.log2((scale ** 2) * (expected_max ** degree)))
            entry["bits_suggested"] = bits_suggested
        entries.append(entry)

    return {
        "bits": bits,
        "signed": signed,
        "scales": entries,
    }
