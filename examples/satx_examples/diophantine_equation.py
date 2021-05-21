# ref: https://link.springer.com/article/10.1007/s00025-021-01366-w

import satx

satx.engine(16)

c = satx.integer()
x = satx.integer()
y = satx.integer()
p = satx.integer()
m = satx.integer()
n = satx.integer()

satx.apply_single([c, x, y, p, m, n], lambda t: t.is_not_in([0, 1]))

assert c * x ** 2 + p ** (2 * m) == 4 * y ** n

while satx.satisfy():
    print(c, x, y, p, m, n)
