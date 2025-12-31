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

from .unit import *


class Rational:
    def __init__(self, x, y):
        self.numerator = x
        self.denominator = y
        if isinstance(self.denominator, Unit):
            assert self.denominator != self.denominator.alu.zero

    def __eq__(self, other):
        c = Unit(self.numerator.alu)
        if isinstance(other, Unit):
            assert self.numerator == c * other
            assert self.denominator == c * self.denominator.alu.one
        else:
            assert self.numerator == c * other.numerator
            assert self.denominator == c * other.denominator
        return True

    def __ne__(self, other):
        assert self.denominator * other.numerator != self.numerator * other.denominator
        return True

    def __neg__(self):
        return Rational(-self.numerator, self.denominator)

    def __add__(self, other):
        return Rational(self.denominator * other.numerator + self.numerator * other.denominator, self.denominator * other.denominator)

    def __radd__(self, other):
        if other == 0:
            return self
        return self + other

    def __rmul__(self, other):
        if other == 1:
            return self
        return self * other

    def __sub__(self, other):
        return Rational(self.denominator * other.numerator - self.numerator * other.denominator, self.denominator * other.denominator)

    def __mul__(self, other):
        if isinstance(other, Unit):
            return Rational(self.numerator * other, self.denominator)
        return Rational(self.numerator * other.numerator, self.denominator * other.denominator)

    def __truediv__(self, other):
        if isinstance(other, Unit):
            return Rational(self.numerator, self.denominator * other)
        if isinstance(other, Rational):
            return self * other.invert()
        return Rational(self.numerator, self.denominator * Unit(self.numerator.alu, value=other))

    def __rtruediv__(self, other):
        if isinstance(other, Unit):
            return Rational(other, other.alu.one) / self
        return Rational(Unit(self.numerator.alu, value=other), self.numerator.alu.one) / self

    def __le__(self, other):
        if isinstance(other, Unit):
            assert self.numerator * other <= self.denominator
        else:
            assert self.numerator * other.denominator <= self.denominator * other.numerator
        return True

    def __ge__(self, other):
        if isinstance(other, Unit):
            assert self.numerator >= other * self.denominator
        else:
            assert self.numerator * other.denominator >= self.denominator * other.numerator if other.numerator != 0 else self.denominator
        return True

    def __lt__(self, other):
        if isinstance(other, Unit):
            assert self.numerator * other < self.denominator
        else:
            assert self.numerator * other.denominator < self.denominator * other.numerator if other.numerator != 0 else self.denominator
        return True

    def __gt__(self, other):
        if isinstance(other, Unit):
            assert self.numerator > other * self.denominator
        else:
            assert self.numerator * other.denominator > self.denominator * other.numerator
        return True

    def __pow__(self, power, modulo=None):
        other = Rational(self.numerator, self.denominator)
        for _ in range(power - 1):
            other *= self
        if modulo is not None:
            return other % modulo
        return other

    def __abs__(self):
        if self.denominator == 0:
            self.denominator = 1
        return Rational(abs(self.numerator), abs(self.denominator))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.denominator == 0:
            self.denominator = 1
        return '({} / {})'.format(self.numerator, self.denominator)

    def __float__(self):
        if self.denominator == 0:
            self.denominator = 1
        return float(self.numerator) / float(self.denominator)

    def invert(self):
        if self.denominator == 0:
            self.denominator = 1
        return Rational(self.denominator, self.numerator)
