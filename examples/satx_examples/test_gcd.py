import math
import satx.gcc

satx.engine(10)

x = satx.integer()
y = satx.integer()
z = satx.integer()

satx.gcc.gcd(x, y, z)

while satx.satisfy():
    print(x, y, z, math.gcd(x.value, y.value))
    if z.value != math.gcd(x.value, y.value):
        raise Exception('ERROR')