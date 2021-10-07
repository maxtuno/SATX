"""
All squares of a board of a specified size (specified numbers of rows and columns) must be colored with the minimum number of colors.
The four corners of any rectangle inside the board must not be assigned the same color.
"""

import itertools

import satx

n, m = 5, 8

opt = 1
while True:
    print('OPTIMAL? : {}'.format(opt))

    satx.engine(opt.bit_length(), cnf_path='aux.cnf')

    # x[i][j] is the color at row i and column j
    x = satx.matrix(dimensions=(n, m))

    # at least one corners of different color for any rectangle inside the board
    for i1, i2 in itertools.combinations(range(n), 2):
        for j1, j2 in itertools.combinations(range(m), 2):
            assert satx.one_of([x[i1][j1], x[i1][j2], x[i2][j1], x[i2][j2]]) != \
                   satx.one_of([x[i1][j1], x[i1][j2], x[i2][j1], x[i2][j2]])

    satx.apply_single(satx.flatten(x), lambda t: t < opt)

    if satx.satisfy('slime'):
        print(x)
        break
    else:
        opt += 1
