"""
See https://en.wikipedia.org/wiki/Dudeney_number

In number theory, a Dudeney number in a given number base b is a natural number
equal to the perfect cube of another natural number such that the digit sum
of the first natural number is equal to the second.
The name derives from Henry Dudeney, who noted the existence of these numbers in one of his puzzles.

There are 5 non trivial numbers for base 10, and the highest such number is formed of 5 digits.
Below, the model is given for base 10.
"""

from math import ceil

import satx

# for base 10
n_digits = 5

satx.engine((10 ** n_digits).bit_length(), cnf_path='aux.cnf')

# n is a (non-trivial) Dudeney number
n = satx.integer()
# s is the perfect cubic root of n
s = satx.integer()
# d[i] is the ith digit of the Dudeney number
d = satx.vector(size=n_digits)

satx.apply_single(d, lambda t: t < 10)

assert 2 <= n < 10 ** n_digits
assert s < ceil((10 ** n_digits) ** (1 / 3)) + 1
assert n == s * s * s
assert sum(d) == s
assert satx.dot(d, [10 ** (n_digits - i - 1) for i in range(n_digits)]) == n

while satx.satisfy('slime'):
    print(n, s, d)
