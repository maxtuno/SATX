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

from decimal import Decimal
from fractions import Fraction
from numbers import Number


class Unit(Number):
    def __init__(self, alu, key=None, block=None, value=None, bits=None, deep=None):
        self.key = key
        self.model = []
        self.block = block
        self.alu = alu
        self.value = None
        self.data = []
        self.bits = bits
        self.deep = deep
        self.bin = []
        if bits is None:
            self.bits = self.alu.bits
            self.deep = [self.bits]
        if deep is not None:
            import functools
            import operator
            self.deep = [deep] if isinstance(deep, int) else deep
            self.bits = functools.reduce(operator.mul, self.deep)
            self.key, self.block = self.alu.create_variable(self.key, self.bits)
            self.data = self.alu.reshape(self.block, self.deep)
        elif block is None and bits is None and value is None:
            self.key, self.block = self.alu.create_variable(self.key)
        elif block is None and bits is not None and value is None:
            self.key, self.block = self.alu.create_variable(self.key, self.bits)
        elif value is not None:
            self.block = self.alu.create_constant(value)
        else:
            self.block = block
        if not self.data:
            self.data = self.block
        if not self.deep:
            self.deep = [self.bits]
        self.key = self.alu.new_key()
        self.alu.mapping(self.key, self.block)

    def is_in(self, item):
        bits = self.alu.int(size=len(item))
        assert sum(self.alu.zero.iff(bits[i], self.alu.one) for i in range(len(item))) == self.alu.one
        assert sum(self.alu.zero.iff(bits[i], item[i]) for i in range(len(item))) == self
        return self

    def is_not_in(self, item):
        for element in item:
            assert self != element
        return self

    def _fixed_info(self, other, op=None):
        raw = getattr(other, "raw", None)
        scale = getattr(other, "scale", None)
        if isinstance(raw, Unit) and isinstance(scale, int):
            if raw.alu is not self.alu:
                if op:
                    raise ValueError(f"cannot {op} values from different SATX engines")
                raise ValueError("cannot operate on values from different SATX engines")
            return raw, scale
        return None

    def __add__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value + other.value
            return self.value + other
        fixed_info = self._fixed_info(other, "+")
        if fixed_info is not None:
            raw, scale = fixed_info
            from .fixed import Fixed
            return Fixed((self * scale) + raw, scale)
        output_block = self.alu.create_block()
        if isinstance(other, Unit):
            self.alu.bv_rca_gate(self.block, other.block, self.alu.true, output_block, None if self.alu.signed else self.alu.true)
            if self.alu.signed:
                msb_eq = self.alu.binary_xnor_gate([self.block[-1], other.block[-1]])
                res_diff = self.alu.binary_xor_gate([output_block[-1], self.block[-1]])
                overflow = self.alu.and_gate([msb_eq, res_diff])
                self.alu.add_block([-overflow])
        else:
            rhs_block = self.alu.create_constant(other)
            self.alu.bv_rca_gate(self.block, rhs_block, self.alu.true, output_block, None if self.alu.signed else self.alu.true)
            if self.alu.signed:
                msb_eq = self.alu.binary_xnor_gate([self.block[-1], rhs_block[-1]])
                res_diff = self.alu.binary_xor_gate([output_block[-1], self.block[-1]])
                overflow = self.alu.and_gate([msb_eq, res_diff])
                self.alu.add_block([-overflow])
        entity = Unit(self.alu, block=output_block)
        self.alu.variables.append(entity)
        return entity

    def __radd__(self, other):
        return self + other

    def __eq__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value == other.value
            else:
                return self.value == other
        fixed_info = self._fixed_info(other, "compare")
        if fixed_info is not None:
            raw, scale = fixed_info
            if self.value is not None and raw.value is not None:
                return (self.value * scale) == raw.value
            scaled = self * scale
            lhs_block = scaled.block if isinstance(scaled, Unit) else self.alu.create_constant(scaled)
            self.alu.bv_eq_gate(lhs_block, raw.block, self.alu.false)
            return self
        if isinstance(other, Unit):
            self.alu.bv_eq_gate(self.block, other.block, self.alu.false)
        else:
            self.alu.bv_eq_gate(self.block, self.alu.create_constant(other), self.alu.false)
        return self

    def __mod__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                if other.value == 0:
                    raise ZeroDivisionError
                return self.value % other.value
            if other == 0:
                raise ZeroDivisionError
            return self.value % other

        other = other if isinstance(other, Unit) else Unit(self.alu, value=other)
        assert other != 0

        if not self.alu.signed:
            output_block = self.alu.create_block()
            self.alu.bv_lur_gate(self.block, other.block, output_block)
            entity = Unit(self.alu, block=output_block)
            self.alu.variables.append(entity)
            return entity

        x_abs = abs(self)
        y_abs = abs(other)
        q0_block = self.alu.create_block()
        r0_block = self.alu.create_block()
        self.alu.bv_lud_gate(x_abs.block, y_abs.block, q0_block, r0_block)
        r0 = Unit(self.alu, block=r0_block)
        self.alu.variables.append(r0)

        rem_nonzero = self.alu.or_gate(r0.block)
        sign_diff = self.alu.binary_xor_gate([self.block[-1], other.block[-1]])
        adjust = self.alu.and_gate([sign_diff, rem_nonzero])
        r_pos = (y_abs - r0).iff(adjust, r0)
        return (-r_pos).iff(other.block[-1], r_pos)

    def __rmod__(self, other):
        if isinstance(other, Unit):
            return other % self
        return Unit(self.alu, value=other) % self

    def __ne__(self, other):
        fixed_info = self._fixed_info(other, "compare")
        if fixed_info is not None:
            raw, scale = fixed_info
            if self.value is not None and raw.value is not None:
                return (self.value * scale) != raw.value
            scaled = self * scale
            lhs_block = scaled.block if isinstance(scaled, Unit) else self.alu.create_constant(scaled)
            return self.alu.bv_eq_gate(lhs_block, raw.block, self.alu.true)
        if isinstance(other, Unit):
            return self.alu.bv_eq_gate(self.block, other.block, self.alu.true)
        return self.alu.bv_eq_gate(self.block, self.alu.create_constant(other), self.alu.true)

    def __mul__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value * other.value
            return self.value * other
        fixed_info = self._fixed_info(other, "*")
        if fixed_info is not None:
            raw, scale = fixed_info
            from .fixed import Fixed
            return Fixed(self * raw, scale)
        if isinstance(other, (float, Decimal, Fraction)) and getattr(self.alu, "default_is_fixed", False):
            from .fixed import Fixed, fixed_const
            const = fixed_const(other, scale=getattr(self.alu, "default_scale", 1))
            if const.raw.alu is not self.alu:
                raise ValueError("cannot multiply values from different SATX engines")
            return Fixed(self * const.raw, const.scale)
        if self.alu.signed:
            def _sext(block, width):
                if len(block) >= width:
                    return block[:width]
                return block + [block[-1]] * (width - len(block))

            width = self.alu.bits
            lhs = _sext(self.block, width)
            if isinstance(other, Unit):
                rhs = _sext(other.block, width)
            else:
                rhs = _sext(self.alu.create_constant(other), width)

            lhs_ext = lhs + [lhs[-1]] * width
            rhs_ext = rhs + [rhs[-1]] * width
            prod_ext = self.alu.create_block(size=2 * width)
            self.alu.bv_pm_gate(lhs_ext, rhs_ext, prod_ext, None)
            result_block = prod_ext[:width]
            sign_bit = result_block[-1]
            for hb in prod_ext[width:]:
                self.alu.bv_eq_gate([hb], [sign_bit], self.alu.false)
            output_block = result_block
        else:
            output_block = self.alu.create_block()
            if isinstance(other, Unit):
                self.alu.bv_pm_gate(self.block, other.block, output_block, self.alu.true)
            else:
                self.alu.bv_pm_gate(self.block, self.alu.create_constant(other), output_block, self.alu.true)
        entity = Unit(self.alu, block=output_block)
        self.alu.variables.append(entity)
        return entity

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power, modulo=None):
        if self.value is not None and not isinstance(power, Unit):
            return self.value ** power
        elif self.value is not None and power.value is not None:
            if modulo is not None:
                return pow(self.value, power.value, modulo)
            return self.value ** power.value
        else:
            if isinstance(power, Unit):
                import functools
                import operator
                assert power > 0
                aa = Unit(self.alu, bits=self.alu.deep)
                assert aa[[0]](0, 1) == 0
                self.alu.variables.append(aa)
                assert functools.reduce(operator.add, [aa[[i]](0, 1) for i in range(1, self.alu.deep)]) == self.alu.one
                assert functools.reduce(operator.add, [aa[[i]](0, i) for i in range(1, self.alu.deep)]) == power
                if modulo is not None:
                    assert modulo != 0
                    return functools.reduce(operator.add, [aa[[i]](0, self ** i) for i in range(1, self.alu.deep)]) % modulo
                return functools.reduce(operator.add, [aa[[i]](0, self ** i) for i in range(1, self.alu.deep)])
            else:
                other = Unit(self.alu, value=1)
                self.alu.variables.append(other)
                for _ in range(power):
                    other *= self
                if modulo is not None:
                    return other % modulo
                return other

    def __rpow__(self, other):
        if isinstance(other, Unit):
            return other ** self
        return Unit(self.alu, value=other) ** self

    def __truediv__(self, other):
        from .rational import Rational
        if isinstance(other, Unit):
            return Rational(self, other)
        return Rational(self, Unit(self.alu, value=other))

    def __rtruediv__(self, other):
        from .rational import Rational
        if isinstance(other, Unit):
            return Rational(other, self)
        return Rational(Unit(self.alu, value=other), self)

    def __floordiv__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                if other.value == 0:
                    raise ZeroDivisionError
                return self.value // other.value
            if other == 0:
                raise ZeroDivisionError
            return self.value // other

        other = other if isinstance(other, Unit) else Unit(self.alu, value=other)
        assert other != 0

        if not self.alu.signed:
            q_block = self.alu.create_block()
            self.alu.bv_lud_gate(self.block, other.block, q_block)
            q = Unit(self.alu, block=q_block)
            self.alu.variables.append(q)
            return q

        x_abs = abs(self)
        y_abs = abs(other)
        q0_block = self.alu.create_block()
        r0_block = self.alu.create_block()
        self.alu.bv_lud_gate(x_abs.block, y_abs.block, q0_block, r0_block)
        q0 = Unit(self.alu, block=q0_block)
        r0 = Unit(self.alu, block=r0_block)
        self.alu.variables.append(q0)
        self.alu.variables.append(r0)

        rem_nonzero = self.alu.or_gate(r0.block)
        sign_diff = self.alu.binary_xor_gate([self.block[-1], other.block[-1]])
        adjust = self.alu.and_gate([sign_diff, rem_nonzero])

        q_trunc = (-q0).iff(sign_diff, q0)
        q_floor = (q_trunc - 1).iff(adjust, q_trunc)
        return q_floor

    def __rfloordiv__(self, other):
        if isinstance(other, Unit):
            return other // self
        return Unit(self.alu, value=other) // self

    def __sub__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value - other.value
            else:
                return self.value - other
        fixed_info = self._fixed_info(other, "-")
        if fixed_info is not None:
            raw, scale = fixed_info
            from .fixed import Fixed
            return Fixed((self * scale) - raw, scale)
        output_block = self.alu.create_block()
        if isinstance(other, Unit):
            output_block = self.alu.bv_rcs_gate(self.block, other.block, output_block)
            if self.alu.signed:
                msb_diff = self.alu.binary_xor_gate([self.block[-1], other.block[-1]])
                res_diff = self.alu.binary_xor_gate([output_block[-1], self.block[-1]])
                overflow = self.alu.and_gate([msb_diff, res_diff])
                self.alu.add_block([-overflow])
        else:
            rhs_block = self.alu.create_constant(other)
            output_block = self.alu.bv_rcs_gate(self.block, rhs_block, output_block)
            if self.alu.signed:
                msb_diff = self.alu.binary_xor_gate([self.block[-1], rhs_block[-1]])
                res_diff = self.alu.binary_xor_gate([output_block[-1], self.block[-1]])
                overflow = self.alu.and_gate([msb_diff, res_diff])
                self.alu.add_block([-overflow])
        entity = Unit(self.alu, block=output_block)
        self.alu.variables.append(entity)
        return entity

    def __rsub__(self, other):
        if isinstance(other, Unit):
            return other - self
        return Unit(self.alu, value=other) - self

    def __lt__(self, other):
        if self.value is not None:
            if isinstance(other, Unit) and other.value is not None:
                return self.value < other.value
            else:
                return self.value < other
        fixed_info = self._fixed_info(other, "compare")
        if fixed_info is not None:
            raw, scale = fixed_info
            if self.value is not None and raw.value is not None:
                return (self.value * scale) < raw.value
            scaled = self * scale
            lhs_block = scaled.block if isinstance(scaled, Unit) else self.alu.create_constant(scaled)
            bvc = self.alu.bv_sle_gate if self.alu.signed else self.alu.bv_ule_gate
            bvc(raw.block, lhs_block, self.alu.true)
            return self
        bvc = self.alu.bv_sle_gate if self.alu.signed else self.alu.bv_ule_gate
        if isinstance(other, Unit):
            bvc(other.block, self.block, self.alu.true)
        else:
            bvc(self.alu.create_constant(other), self.block, self.alu.true)
        return self

    def __le__(self, other):
        if self.value is not None:
            if isinstance(other, Unit) and other.value is not None:
                return self.value <= other.value
            else:
                return self.value <= other
        fixed_info = self._fixed_info(other, "compare")
        if fixed_info is not None:
            raw, scale = fixed_info
            if self.value is not None and raw.value is not None:
                return (self.value * scale) <= raw.value
            scaled = self * scale
            lhs_block = scaled.block if isinstance(scaled, Unit) else self.alu.create_constant(scaled)
            bvc = self.alu.bv_sle_gate if self.alu.signed else self.alu.bv_ule_gate
            bvc(lhs_block, raw.block, self.alu.false)
            return self
        bvc = self.alu.bv_sle_gate if self.alu.signed else self.alu.bv_ule_gate
        if isinstance(other, Unit):
            bvc(self.block, other.block, self.alu.false)
        else:
            bvc(self.block, self.alu.create_constant(other), self.alu.false)
        return self

    def __gt__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value > other.value
            else:
                return self.value > other
        fixed_info = self._fixed_info(other, "compare")
        if fixed_info is not None:
            raw, scale = fixed_info
            if self.value is not None and raw.value is not None:
                return (self.value * scale) > raw.value
            scaled = self * scale
            lhs_block = scaled.block if isinstance(scaled, Unit) else self.alu.create_constant(scaled)
            bvc = self.alu.bv_sle_gate if self.alu.signed else self.alu.bv_ule_gate
            bvc(lhs_block, raw.block, self.alu.true)
            return self
        bvc = self.alu.bv_sle_gate if self.alu.signed else self.alu.bv_ule_gate
        if isinstance(other, Unit):
            bvc(self.block, other.block, self.alu.true)
        else:
            bvc(self.block, self.alu.create_constant(other), self.alu.true)
        return self

    def __ge__(self, other):
        if self.value is not None:
            if isinstance(other, Unit) and other.value is not None:
                return self.value >= other.value
            else:
                return self.value >= other
        fixed_info = self._fixed_info(other, "compare")
        if fixed_info is not None:
            raw, scale = fixed_info
            if self.value is not None and raw.value is not None:
                return (self.value * scale) >= raw.value
            scaled = self * scale
            lhs_block = scaled.block if isinstance(scaled, Unit) else self.alu.create_constant(scaled)
            bvc = self.alu.bv_sle_gate if self.alu.signed else self.alu.bv_ule_gate
            bvc(raw.block, lhs_block, self.alu.false)
            return self
        bvc = self.alu.bv_sle_gate if self.alu.signed else self.alu.bv_ule_gate
        if isinstance(other, Unit):
            bvc(other.block, self.block, self.alu.false)
        else:
            bvc(self.alu.create_constant(other), self.block, self.alu.false)
        return self

    def __neg__(self):
        if self.value is not None:
            return -self.value
        #if self.alu.signed:
        #    entity = Unit(self.alu, block=[-b for b in self.block]) + self.alu.one
        #    self.alu.variables.append(entity)
        return self.alu.zero - self

    def __abs__(self):
        if self.value is not None:
            return abs(self.value)
        if self.alu.signed:
            neg = -self
            output_block = self.alu.bv_mux_gate(neg.block, self.block, self.block[-1])
            entity = Unit(self.alu, block=output_block)
            self.alu.variables.append(entity)
            assert entity >= 0
            return entity
        lst = [-self, self]
        bits = self.alu.int(size=len(lst))
        assert sum(self.alu.zero.iff(bits[i], self.alu.one) for i in range(len(lst))) == self.alu.one
        return sum(self.alu.zero.iff(bits[i], lst[i]) for i in range(len(lst)))

    def __and__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value & other.value
            else:
                return self.value & other
        if isinstance(other, Unit):
            output_block = self.alu.bv_and_gate(self.block, other.block)
        else:
            output_block = self.alu.bv_and_gate(self.block, self.alu.create_constant(other))
        entity = Unit(self.alu, block=output_block)
        self.alu.variables.append(entity)
        return entity

    def __or__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value | other.value
            else:
                return self.value | other
        if isinstance(other, Unit):
            output_block = self.alu.bv_or_gate(self.block, other.block)
        else:
            output_block = self.alu.bv_or_gate(self.block, self.alu.create_constant(other))
        entity = Unit(self.alu, block=output_block)
        self.alu.variables.append(entity)
        return entity

    def __xor__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value ^ other.value
            else:
                return self.value ^ other
        if isinstance(other, Unit):
            output_block = self.alu.bv_xor_gate(self.block, other.block)
        else:
            output_block = self.alu.bv_xor_gate(self.block, self.alu.create_constant(other))
        entity = Unit(self.alu, block=output_block)
        self.alu.variables.append(entity)
        return entity

    def __lshift__(self, other):
        if isinstance(other, Unit):
            assert 0 < other
        y = 2 * other
        x = self * y
        return x

    def __rshift__(self, other):
        if isinstance(other, Unit):
            assert 0 < other
        y = 2 * other
        x = self // y
        return x

    def iff(self, bit, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value if bit else other.value
            else:
                return self.value if bit else other
        if isinstance(bit, Unit):
            import functools
            import operator
            if isinstance(other, Unit):
                return self.iff(functools.reduce(operator.and_, [self.alu.zero.iff(bit[j], self.alu.one) for j in range(self.alu.bits)])[0], other)
            else:
                return self.iff(functools.reduce(operator.and_, [self.alu.zero.iff(bit[j], self.alu.one) for j in range(self.alu.bits)])[0], self.alu.create_constant(other))
        if isinstance(other, Unit):
            output_block = self.alu.bv_mux_gate(self.block, other.block, bit)
            entity = Unit(self.alu, block=output_block)
            self.alu.variables.append(entity)
            return entity
        else:
            output_block = self.alu.bv_mux_gate(self.block, self.alu.create_constant(other), bit)
            entity = Unit(self.alu, block=output_block)
            self.alu.variables.append(entity)
            return entity

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[item]
        bb = self.data[:]
        for i in item:
            bb = bb[i]
        return lambda a, b: (a if isinstance(a, Unit) else self.alu.int(value=a)).iff(-bb, (b if isinstance(b, Unit) else self.alu.int(value=b)))

    @property
    def binary(self):
        def __encode(n):
            if self.bin:
                return self.bin
            bits = []
            for i in range(self.bits):
                if n % 2 == 0:
                    bits += [False]
                else:
                    bits += [True]
                n //= 2
            self.bin = bits
            return bits

        return self.alu.reshape(__encode(self.value), self.deep)

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.__repr__())

    def __int__(self):
        return self.value

    def __float__(self):
        return float(self.value)

    def clear(self):
        self.value = None

    def reverse(self, copy=False):
        if copy:
            entity = Unit(self.alu, block=self.block[::-1])
            self.alu.variables.append(entity)
            return entity
        else:
            self.block = self.block[::-1]
        return self
