"""
See https://en.wikipedia.org/wiki/Verbal_arithmetic

A model for a general form of this problem is in CryptoPuzzle.py
"""

import satx

satx.engine(16, cnf_path='aux.cnf')

# letters[i] is the digit of the ith letter involved in the equation
s, e, n, d, m, o, r, y = letters = satx.vector(size=8)

satx.apply_single(letters, lambda t: t < 10)

# letters are given different values
satx.all_different(letters),

# words cannot start with 0
assert s > 0
assert m > 0

# respecting the mathematical equation
assert satx.dot([s, e, n, d], [1000, 100, 10, 1]) + satx.dot([m, o, r, e], [1000, 100, 10, 1]) == satx.dot([m, o, n, e, y], [10000, 1000, 100, 10, 1])

if satx.satisfy('slime'):
    print(letters)
else:
    print('Infeasible...')
