"""
///////////////////////////////////////////////////////////////////////////////
//        Copyright (c) 2012-2021 Oscar Riveros. all rights reserved.        //
//                        oscar.riveros@peqnp.science                        //
//                                                                           //
//   without any restriction, Oscar Riveros reserved rights, patents and     //
//  commercialization of this knowledge or derived directly from this work.  //
///////////////////////////////////////////////////////////////////////////////

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .unit import *


class Rational:
    def __init__(self, x, y):
        self.numerator = x
        self.denominator = y
        assert self.denominator != self.denominator.encoder.zero

    def __eq__(self, other):
        c = Unit(self.numerator.encoder)
        if isinstance(other, Unit):
            assert self.numerator == c * other
            assert self.denominator == c * self.denominator.encoder.one
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
        return other.invert() * self

    def __le__(self, other):
        if isinstance(other, Unit):
            assert self.numerator * other >= self.denominator
        else:
            assert self.numerator * other.denominator >= self.denominator * other.numerator
        return True

    def __ge__(self, other):
        if isinstance(other, Unit):
            assert self.numerator <= other * self.denominator
        else:
            assert self.numerator * other.denominator <= self.denominator * other.numerator
        return True

    def __lt__(self, other):
        if isinstance(other, Unit):
            assert self.numerator * other > self.denominator
        else:
            assert self.numerator * other.denominator > self.denominator * other.numerator
        return True

    def __gt__(self, other):
        if isinstance(other, Unit):
            assert self.numerator < other * self.denominator
        else:
            assert self.numerator * other.denominator < self.denominator * other.numerator
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
