import numpy
import satx

n = 3

satx.engine(8)

c = -1

xs = satx.matrix(dimensions=(n, n))

satx.all_different(satx.flatten(xs))

for i in range(n):
    assert sum(xs[i][j] for j in range(n)) == c
for j in range(n):
    assert sum(xs[i][j] for i in range(n)) == c 

assert sum(xs[i][i] for i in range(n)) == c
assert sum(xs[i][n - 1 - i] for i in range(n)) == c

if satx.satisfy(turbo=True, log=True):
    print(c)
    print(numpy.vectorize(int)(xs))
else:
    print('Infeasible...')
