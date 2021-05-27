import sys

import numpy as np


def pieces(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


hs = sys.argv[1]
hs = (64 - len(hs)) * '0' + hs

binary = ''
for dig in pieces(hs, 8):
    binary += np.binary_repr(int('0x' + str(dig), 16), 32)[::-1]

lit = 2049 - 1
for c in binary:
    if c == '1':
        print(str(+(1 + lit)) + ' 0')
    else:
        print(str(-(1 + lit)) + ' 0')
    lit += 1
