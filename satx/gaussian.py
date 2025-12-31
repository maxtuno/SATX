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


class Gaussian:
    def __init__(self, x, y):
        self.real = x
        self.imag = y

    def __eq__(self, other):
        assert self.real == other.real
        assert self.imag == other.imag
        return True

    def __ne__(self, other):
        bit = Unit(self.real.alu, bits=2)
        assert (self.real - other.real).iff(bit[0], self.imag - other.imag) != 0
        return True

    def __neg__(self):
        return Gaussian(-self.real, -self.imag)

    def __add__(self, other):
        return Gaussian(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return Gaussian(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        return Gaussian((self.real * other.real) - (self.imag * other.imag), ((self.real * other.imag) + (self.imag * other.real)))

    def __truediv__(self, other):
        return Gaussian(
            ((self.real * other.real) + (self.imag * other.imag)) / (other.real ** 2 + other.imag ** 2), ((self.imag * other.real) - (self.real * other.imag)) / (other.real ** 2 + other.imag ** 2))

    def __pow__(self, power, modulo=None):
        other = self
        for _ in range(power - 1):
            other *= self
        return other

    def __abs__(self):
        return Gaussian(self.real.alu.sqrt(self.real ** 2 + self.imag ** 2), 0)

    def __repr__(self):
        return '({}+{}j)'.format(self.real, self.imag)

    def __str__(self):
        return str(self.__repr__())

    def __complex__(self):
        return complex(int(self.real), int(self.imag))

    def conjugate(self):
        return Gaussian(self.real, -self.imag)
