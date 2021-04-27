import numpy
import satx

n = 5
m = n ** 2

satx.engine(m.bit_length(), cnf='knights_tour.cnf')

r = satx.vector(size=m)
c = satx.vector(size=m)
rr = satx.vector(size=m)
cc = satx.vector(size=m)

satx.apply_single(r + c, lambda x: 1 <= x <= n)
satx.apply_single(rr + cc, lambda x: x.is_in([1, 2]))
satx.all_different([r[i] + n * c[i] for i in range(m)])

for i in range(m - 1):
    assert c[i + 1] == satx.one_of([c[i] + cc[i], c[i] - cc[i]])
    assert r[i + 1] == satx.one_of([r[i] + rr[i], r[i] - rr[i]])
    assert cc[i] != rr[i]

if satx.external_satisfy(solver='java -jar -Xmx4g blue.jar'):
    c = numpy.vectorize(int)(c) - 1
    r = numpy.vectorize(int)(r) - 1
    t = numpy.zeros(shape=(n, n), dtype=int)
    for k, (i, j) in enumerate(zip(c, r)):
        t[i][j] = k + 1
    print(t)
else:
    print('Infeasible ...')
