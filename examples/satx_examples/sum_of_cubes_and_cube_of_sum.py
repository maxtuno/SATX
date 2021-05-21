# ref: https://arxiv.org/abs/2012.04139

import satx

satx.engine(16)

a = satx.integer()
x = satx.integer()
y = satx.integer()
z = satx.integer()

satx.apply_single([a, x, y, z], lambda t: t != 0)

assert a * (x ** 3 + y ** 3 + z ** 3) == (x + y + z) ** 3

while satx.satisfy():
    print(a, x, y, z)
