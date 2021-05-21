import numpy as np
import satx

satx.engine(10)

x = satx.integer()
y = satx.integer()

assert satx.rotate(x, -1) == y

while satx.satisfy():
    print(np.vectorize(int)(x.binary))
    print(np.vectorize(int)(y.binary))
    print()

print(80 * '-')

import numpy as np
import satx

satx.engine(10)

x = satx.integer()
y = satx.integer()

assert satx.rotate(x, 1) == y

while satx.satisfy():
    print(np.vectorize(int)(x.binary))
    print(np.vectorize(int)(y.binary))
    print()