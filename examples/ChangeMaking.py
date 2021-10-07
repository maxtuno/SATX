"""
See https://en.wikipedia.org/wiki/Change-making_problem
"""

import satx

k = 13
coins = [1, 5, 10, 20, 50, 100, 200]

opt = k
while True:
    satx.engine(sum(coins).bit_length(), cnf_path='aux.cnf')

    x = satx.vector(size=len(coins))

    assert satx.dot(x, coins) == k

    assert sum(x) < opt

    if satx.satisfy('slime'):
        opt = sum(x)
        print(opt, x)
    else:
        break


