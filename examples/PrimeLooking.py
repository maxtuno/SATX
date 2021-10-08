"""
See Model in OscaR

Martin Gardner Problem:
 * Call a number "prime-looking" if it is composite but not divisible by 2,3 or 5.
 * The three smallest prime-looking numbers are 49, 77 and 91.
 * There are 168 prime numbers less than 1000.
 * How many prime-looking numbers are there less than 1000?
"""

import satx

satx.engine(10, cnf_path='aux.cnf')

# the number we look for
x = satx.integer()
# a first divider
d1 = satx.integer()
# a second divider
d2 = satx.integer()

assert x < 1000
assert 2 <= d1 < 1000
assert 2 <= d2 < 1000

assert x == d1 * d2
assert x % 2 != 0
assert x % 3 != 0
assert x % 5 != 0
assert d1 <= d2

while satx.satisfy('kissat'):
    print(x, d1, d2)
