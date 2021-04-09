# ref: https://projecteuclid.org/journals/proceedings-of-the-japan-academy-series-a-mathematical-sciences/volume-80/issue-4/A-note-on-the-exponential-diophantine-equation-ax--by/10.3792/pjaa.80.21.full

import satx

satx.engine(16)

a = satx.integer()
b = satx.integer()
c = satx.integer()
x = satx.integer()
y = satx.integer()
z = satx.integer()

satx.apply_single([a, b, c, x, y, z], lambda t: t.is_not_in([0, 1]))

assert a ** x + b ** y == c ** z

while satx.satisfy():
    print(a, b, c, x, y, z)