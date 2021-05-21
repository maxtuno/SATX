# ref : http://polip.zib.de/maxcut/sched-6-4711.php

import satx

opt, sol = 0, []
while True:
    satx.engine(10)

    b2 = satx.integer()
    b3 = satx.integer()
    b4 = satx.integer()
    b5 = satx.integer()
    b6 = satx.integer()
    b7 = satx.integer()
    b8 = satx.integer()
    b9 = satx.integer()
    b10 = satx.integer()
    b11 = satx.integer()
    b12 = satx.integer()
    b13 = satx.integer()
    b14 = satx.integer()
    b15 = satx.integer()
    b16 = satx.integer()
    x17 = satx.integer()

    satx.all_binaries([b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16])

    assert 2 * b13 * b8 - 2 * b8 - 2 * b13 * b4 + 2 * b4 + 2 * b8 * b2 - 2 * b2 + 2 * b14 * b8 + 2 * b4 * b2 - 2 * b15 * b4 - 2 * b6 * b5 + 2 * b5 - 2 * b6 + 2 * b9 * b6 - 2 * b9 + 2 * b6 * b3 - 2 * b3 + 2 * b11 * b3 - 4 * b11 - 2 * b5 * b4 - 2 * b16 * b5 - 2 * b13 * b10 - 2 * b10 + 2 * b10 * b5 + 2 * b7 * b6 - 2 * b7 + 2 * b10 * b7 - 2 * b16 * b14 + 2 * b16 * b15 - 2 * b9 * b8 + 2 * b16 * b9 + 2 * b13 * b11 + 2 * b11 * b9 + 2 * b12 * b10 - 2 * b12 + 2 * b12 * b11 <= -x17

    assert x17 > opt

    if satx.satisfy(turbo=True):
        opt = x17.value
        sol = satx.values([b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16])
        print(opt, sol)
    else:
        print(60 * '-')
        for i, x in enumerate(sol):
            print('x{}\t{}'.format(i, x))
        break
