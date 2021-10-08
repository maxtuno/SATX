"""
Dad wants one-cent, two-cent, three-cent, five-cent, and ten-cent stamps.
He said to get four each of two sorts and three each of the others, but I've
forgotten which. He gave me exactly enough to buy them; just these dimes."
How many stamps of each type does Dad want? A dime is worth ten cents.
-- J.A.H. Hunter
"""

import satx

satx.engine(10, cnf_path='aux.cnf')

# x is the number of dimes
x = satx.integer()

# s[i] is the number of stamps of value 1, 2, 3, 5 and 10 according to i
s = satx.vector(size=5)

satx.apply_single(s, lambda t: t.is_in([3, 4]))

# 26 is a safe upper bound
assert x <= 26


assert satx.dot(s, [1, 2, 3, 5, 10]) == x * 10

while satx.satisfy('slime'):
    print(s, x)
