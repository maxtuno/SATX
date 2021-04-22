# https://www.sat-x.io/2021/04/22/international-math-olympiad-2006-question-4/

import satx

satx.engine(20)

_2 = satx.constant(2)

x = satx.integer()
y = satx.integer()

Y = satx.one_of([y, -y])

assert 1 + _2 ** x + _2 ** (2 * x + 1) == Y ** 2

while satx.satisfy():
    print(x, y)
