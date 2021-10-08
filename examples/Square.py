"""
See http://en.wikibooks.org/wiki/Puzzles/Arithmetical_puzzles/Digits_of_the_Square

There is one four-digit whole number x, such that the last four digits of x^2
are in fact the original number x. What is it?
"""

import satx

satx.engine(30, cnf_path='aux.cnf')

# x is the number we look for
x = satx.integer()

# d[i] is the ith digit of x
d = satx.vector(size=4)

satx.apply_single(d, lambda t: t.is_in(range(10)))

assert 1000 <= x < 10000
assert satx.dot(d, [1000, 100, 10, 1]) == x
assert (x * x) % 10000 == x

if satx.satisfy('slime'):
    print(x, d)
else:
    print('Infeasible...')
