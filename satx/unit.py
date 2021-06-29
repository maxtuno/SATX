"""
Copyright (c) 2012-2021 Oscar Riveros [oscar.riveros@peqnp.science].

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

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

    def __add__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value + other.value
            return self.value + other
        output_block = self.alu.create_block()
        if isinstance(other, Unit):
            self.alu.bv_rca_gate(self.block, other.block, self.alu.true, output_block, self.alu.true)
        else:
            if other < 0:
                self.alu.bv_rcs_gate(self.block, self.alu.create_constant(-other), output_block)
            else:
                self.alu.bv_rca_gate(self.block, self.alu.create_constant(other), self.alu.true, output_block, self.alu.true)
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
        if isinstance(other, Unit):
            self.alu.bv_eq_gate(self.block, other.block, self.alu.false)
        else:
            self.alu.bv_eq_gate(self.block, self.alu.create_constant(other), self.alu.false)
        return self

    def __mod__(self, other):
        if self.value is not None and other.value is not None:
            return self.value % other.value
        if self.value is not None and other.value is None:
            return self.value % other
        output_block = self.alu.create_block()
        if isinstance(other, Unit):
            self.alu.bv_lur_gate(self.block, other.block, output_block)
        else:
            self.alu.bv_lur_gate(self.block, self.alu.create_constant(other), output_block)
        entity = Unit(self.alu, block=output_block)
        self.alu.variables.append(entity)
        return entity

    def __ne__(self, other):
        if isinstance(other, Unit):
            return self.alu.bv_eq_gate(self.block, other.block, self.alu.true)
        return self.alu.bv_eq_gate(self.block, self.alu.create_constant(other), self.alu.true)

    def __mul__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value * other.value
            return self.value * other
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
                aa = Unit(self.alu, bits=2 * self.bits // 3)
                self.alu.variables.append(aa)
                assert functools.reduce(operator.add, [aa[[i]](0, 1) for i in range(2 * self.bits // 3)]) == self.alu.one
                assert functools.reduce(operator.ior, [aa[[i]](0, i) for i in range(2 * self.bits // 3)]) == power
                if modulo is not None:
                    assert modulo != 0
                    return functools.reduce(operator.ior, [aa[[i]](0, self ** i) for i in range(2 * self.bits // 3)]) % modulo
                return functools.reduce(operator.ior, [aa[[i]](0, self ** i) for i in range(2 * self.bits // 3)])
            else:
                other = Unit(self.alu, value=1)
                self.alu.variables.append(other)
                for _ in range(power):
                    other *= self
                if modulo is not None:
                    return other % modulo
                return other

    def __truediv__(self, other):
        if self.value is not None:
            if isinstance(other, Unit) and other.value is not None:
                if other.value == 0:
                    from math import nan
                    return nan
                return self.value / other.value
        output_block = self.alu.create_block()
        if isinstance(other, Unit):
            self.alu.bv_lud_gate(self.block, other.block, output_block, self.alu.zero.block)
        else:
            self.alu.bv_lud_gate(self.block, self.alu.create_constant(other), output_block, self.alu.zero.block)
        entity = Unit(self.alu, block=output_block)
        self.alu.variables.append(entity)
        return entity

    def __sub__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value - other.value
            else:
                return self.value - other
        output_block = self.alu.create_block()
        if isinstance(other, Unit):
            output_block = self.alu.bv_rcs_gate(self.block, other.block, output_block)
        else:
            output_block = self.alu.bv_rcs_gate(self.block, self.alu.create_constant(other), output_block)
        entity = Unit(self.alu, block=output_block)
        self.alu.variables.append(entity)
        return entity

    def __rsub__(self, other):
        return -(self - other)

    def __lt__(self, other):
        if self.value is not None:
            if isinstance(other, Unit) and other.value is not None:
                return self.value < other.value
            else:
                return self.value < other
        if isinstance(other, Unit):
            self.alu.bv_ule_gate(other.block, self.block, self.alu.true)
        else:
            self.alu.bv_ule_gate(self.alu.create_constant(other), self.block, self.alu.true)
        return self

    def __le__(self, other):
        return self.__lt__(other + 1)

    def __gt__(self, other):
        if self.value is not None:
            if isinstance(other, Unit):
                return self.value > other.value
            else:
                return self.value > other
        if isinstance(other, Unit):
            self.alu.bv_ule_gate(self.block, other.block, self.alu.true)
        else:
            self.alu.bv_ule_gate(self.block, self.alu.create_constant(other), self.alu.true)
        return self

    def __ge__(self, other):
        if other > 0:
            return self.__gt__(other - 1)
        else:
            return self

    def __neg__(self):
        if self.value is not None:
            return -self.value
        entity = Unit(self.alu, block=[-b for b in self.block]) + self.alu.one
        self.alu.variables.append(entity)
        return entity

    def __abs__(self):
        if self.value is not None:
            return abs(self.value)
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
        x = self / y
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
