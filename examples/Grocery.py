"""
See "Constraint Programming in Oz. A Tutorial" by C. Schulte and G. Smolka, 2001

A kid goes into a grocery store and buys four items.
The cashier charges $7.11, the kid pays and is about to leave when the cashier calls the kid back, and says
``Hold on, I multiplied the four items instead of adding them;
  I'll try again;
  Hah, with adding them the price still comes to $7.11''.
What were the prices of the four items?
"""

import satx

# 711 * 100 * 100 * 100 -> 30 bits
satx.engine(30, cnf_path='aux.cnf')

# x[i] is the price (multiplied by 100) of the ith item
x = satx.vector(size=4)

satx.apply_single(x, lambda t: t < 711)

# adding the prices of items corresponds to 711 cents
assert sum(x) == 711

# multiplying the prices of items corresponds to 711 cents (times 1000000)
assert x[0] * x[1] * x[2] * x[3] == 711 * 100 * 100 * 100

if satx.satisfy('slime'):
    print(x)
else:
    print('Infeasible...')
