"""
Copyright (c) 2012–2026 Oscar Riveros

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

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
from typing import Any, List, Optional, Tuple, Union
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


def _parse_bounds_assumptions(assumptions: Any) -> Tuple[Optional[Fraction], Optional[Fraction]]:
    if assumptions is None:
        return None, None
    if isinstance(assumptions, dict):
        min_value = assumptions.get("min", None)
        max_value = assumptions.get("max", None)
    elif isinstance(assumptions, (tuple, list)) and len(assumptions) == 2:
        min_value, max_value = assumptions
    else:
        raise TypeError("assumptions must be a dict with min/max or a (min, max) tuple")
    if min_value is not None:
        if isinstance(min_value, bool) or not _is_numeric_value(min_value):
            raise TypeError("assumptions min must be numeric or None")
        min_value = _coerce_numeric_value(min_value)
    if max_value is not None:
        if isinstance(max_value, bool) or not _is_numeric_value(max_value):
            raise TypeError("assumptions max must be numeric or None")
        max_value = _coerce_numeric_value(max_value)
    return min_value, max_value


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


def _fixed_mul_cache(alu: Any) -> dict:
    cache = getattr(alu, "_fixed_mul_cache", None)
    if cache is None:
        cache = {}
        alu._fixed_mul_cache = cache
    return cache


def _fixed_mul_cache_key(lhs: Unit, rhs: Unit, scale: int, mode: str) -> tuple:
    lhs_block = tuple(lhs.block)
    rhs_block = tuple(rhs.block)
    if lhs_block > rhs_block:
        lhs_block, rhs_block = rhs_block, lhs_block
    return (lhs_block, rhs_block, lhs.bits, scale, lhs.alu.signed, mode)


def _cached_raw_product(lhs: Unit, rhs: Unit) -> Any:
    cache = _fixed_mul_cache(lhs.alu)
    key = _fixed_mul_cache_key(lhs, rhs, 0, "raw")
    cached = cache.get(key)
    if cached is not None:
        return cached
    prod = lhs * rhs
    cache[key] = prod
    return prod


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

        cache = _fixed_mul_cache(self.raw.alu)
        key = _fixed_mul_cache_key(self.raw, other.raw, self.scale, "exact")
        cached = cache.get(key)
        if cached is not None:
            return cached
        r = Unit(self.raw.alu, bits=self.raw.bits)
        self.raw.alu.variables.append(r)
        prod = _cached_raw_product(self.raw, other.raw)
        assert prod == (r * self.scale)
        result = Fixed(r, self.scale)
        cache[key] = result
        return result

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


def fixed_lcm_scale(a: Fixed, b: Fixed) -> Tuple[int, int, int]:
    if not isinstance(a, Fixed) or not isinstance(b, Fixed):
        raise TypeError("fixed_lcm_scale expects (Fixed, Fixed)")
    if a.raw.alu is not b.raw.alu:
        raise ValueError("cannot combine Fixed values from different SATX engines")
    gcd = math.gcd(a.scale, b.scale)
    lcm = (a.scale // gcd) * b.scale
    return lcm, lcm // a.scale, lcm // b.scale


def fixed_rescale_to(x: Fixed, target_scale: int) -> Fixed:
    if not isinstance(x, Fixed):
        raise TypeError("fixed_rescale_to expects Fixed")
    target_scale = _require_positive_int("target_scale", target_scale)
    if x.scale == target_scale:
        return x
    _check_scale_fits_unit(target_scale, x.raw)

    if target_scale % x.scale == 0:
        factor = target_scale // x.scale
        if x.raw.value is not None:
            raw_value = x.raw.value * factor
            _check_raw_value_fits_engine(raw_value)
            return Fixed(_stdlib.constant(raw_value), target_scale)
        raw = x.raw * factor
        if isinstance(raw, int):
            _check_raw_value_fits_engine(raw)
            return Fixed(_stdlib.constant(raw), target_scale)
        return Fixed(raw, target_scale)

    if x.scale % target_scale == 0:
        factor = x.scale // target_scale
        if x.raw.value is not None:
            if x.raw.value % factor != 0:
                raise ValueError("fixed_rescale_to requires exact division")
            raw_value = x.raw.value // factor
            _check_raw_value_fits_engine(raw_value)
            return Fixed(_stdlib.constant(raw_value), target_scale)
        assert x.raw % factor == 0
        raw = x.raw // factor
        if isinstance(raw, int):
            _check_raw_value_fits_engine(raw)
            return Fixed(_stdlib.constant(raw), target_scale)
        return Fixed(raw, target_scale)

    raise ValueError("target_scale must be a multiple of the source scale (or vice versa)")


def fixed_add_rescaled(a: Fixed, b: Fixed, *, target_scale: Optional[int] = None) -> Fixed:
    if not isinstance(a, Fixed) or not isinstance(b, Fixed):
        raise TypeError("fixed_add_rescaled expects (Fixed, Fixed)")
    if a.raw.alu is not b.raw.alu:
        raise ValueError("cannot add Fixed values from different SATX engines")

    if target_scale is None:
        target_scale, _, _ = fixed_lcm_scale(a, b)
    else:
        target_scale = _require_positive_int("target_scale", target_scale)

    ar = fixed_rescale_to(a, target_scale)
    br = fixed_rescale_to(b, target_scale)
    if ar.raw.value is not None and br.raw.value is not None:
        raw_value = ar.raw.value + br.raw.value
        _check_raw_value_fits_engine(raw_value)
        return Fixed(_stdlib.constant(raw_value), target_scale)
    raw_sum = ar.raw + br.raw
    if isinstance(raw_sum, int):
        _check_raw_value_fits_engine(raw_sum)
        return Fixed(_stdlib.constant(raw_sum), target_scale)
    return Fixed(raw_sum, target_scale)


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
    prod = _cached_raw_product(a.raw, b.raw)
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
    prod = _cached_raw_product(a.raw, b.raw)
    if isinstance(prod, int):
        raw_value = _as_int_from_value(prod, a.scale, "round")
        _check_raw_value_fits_engine(raw_value)
        return Fixed(_stdlib.constant(raw_value), a.scale)
    raw = as_int(Fixed(prod, a.scale), policy="round")
    if isinstance(raw, Unit):
        return Fixed(raw, a.scale)
    _check_raw_value_fits_engine(raw)
    return Fixed(_stdlib.constant(raw), a.scale)


def fixed_bounds(expr_or_fixed: Any, assumptions: Any = None) -> Tuple[Fraction, Fraction]:
    if isinstance(expr_or_fixed, Fixed):
        raw = expr_or_fixed.raw
        scale = expr_or_fixed.scale
    elif isinstance(expr_or_fixed, Unit):
        raw = expr_or_fixed
        scale = 1
    elif _is_numeric_value(expr_or_fixed):
        value = _coerce_numeric_value(expr_or_fixed)
        return value, value
    else:
        raise TypeError("fixed_bounds expects Fixed, Unit, or numeric value")

    bits = raw.bits if raw.bits is not None else raw.alu.bits
    signed = raw.alu.signed
    if raw.value is not None:
        min_raw = max_raw = raw.value
    else:
        if signed:
            min_raw = -(1 << (bits - 1))
            max_raw = (1 << (bits - 1)) - 1
        else:
            min_raw = 0
            max_raw = (1 << bits) - 1

    min_value = Fraction(min_raw, scale)
    max_value = Fraction(max_raw, scale)
    min_assumed, max_assumed = _parse_bounds_assumptions(assumptions)
    if min_assumed is not None:
        min_value = max(min_value, min_assumed)
    if max_assumed is not None:
        max_value = min(max_value, max_assumed)
    if min_value > max_value:
        raise ValueError("fixed_bounds assumptions are inconsistent with engine range")
    return min_value, max_value


def fixed_advice(*vars: Any, degree: int = 2, expected_max: Optional[float] = None, explain: bool = False) -> dict:
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
    if not isinstance(explain, bool):
        raise TypeError("explain must be a bool")

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

    result = {
        "bits": bits,
        "signed": signed,
        "scales": entries,
    }
    if explain:
        scale_list = ", ".join(str(entry["scale"]) for entry in entries)
        lines = [
            f"raw_max_abs={raw_max_abs} for {bits}-bit {'signed' if signed else 'unsigned'} engine.",
            f"Scales: {scale_list}; resolution=1/scale, value_max_abs=raw_max_abs/scale.",
            f"safe_mul_value_max~sqrt(raw_max_abs)/scale; safe_pow_value_max uses degree={degree}.",
        ]
        if any(entry["scale"] * entry["scale"] > raw_max_abs for entry in entries):
            lines.append("Overflow risk: scale^2 exceeds raw_max_abs for some scales.")
        if expected_max is not None:
            lines.append(f"bits_suggested uses expected_max={expected_max}.")
        result["explain"] = "\n".join(lines[:6])
    return result


def _unit_bits_from_value(u: Unit) -> List[int]:
    if u.value is None:
        raise ValueError("variable has no model value; call satx.satisfy() first")
    bits = u.bits if u.bits is not None else u.alu.bits
    value = int(u.value)
    if u.alu.signed and value < 0:
        value = (1 << bits) + value
    out = []
    for _ in range(bits):
        out.append(value & 1)
        value >>= 1
    return out


def block_fixed_solution(vars: Any) -> List[int]:
    _stdlib.check_engine()
    if isinstance(vars, (Fixed, Unit)):
        vars = [vars]
    if not isinstance(vars, (list, tuple)):
        raise TypeError("block_fixed_solution expects a list of Fixed or Unit")
    clause = []
    for var in vars:
        if isinstance(var, Fixed):
            raw = var.raw
        elif isinstance(var, Unit):
            raw = var
        else:
            raise TypeError("block_fixed_solution expects Fixed or Unit entries")
        bits = _unit_bits_from_value(raw)
        for bit_id, bit_val in zip(raw.block, bits):
            clause.append(-bit_id if bit_val else bit_id)
    if not clause:
        raise ValueError("block_fixed_solution requires at least one variable")
    _stdlib.csp.add_block(clause)
    return clause


def enumerate_models_fixed(
    vars: Any,
    *,
    limit: Optional[int] = None,
    solver: str = "slime",
    params: str = "",
    log: bool = False,
):
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be a Python int or None")
        if limit <= 0:
            raise ValueError("limit must be > 0")
    if isinstance(vars, (Fixed, Unit)):
        vars = [vars]
    if not isinstance(vars, (list, tuple)):
        raise TypeError("enumerate_models_fixed expects a list of Fixed or Unit")

    count = 0
    while _stdlib.satisfy(solver=solver, params=params, log=log):
        model = []
        for var in vars:
            if isinstance(var, Fixed):
                model.append(var.value)
            elif isinstance(var, Unit):
                model.append(var.value)
            else:
                raise TypeError("enumerate_models_fixed expects Fixed or Unit entries")
        yield model
        block_fixed_solution(vars)
        count += 1
        if limit is not None and count >= limit:
            break
