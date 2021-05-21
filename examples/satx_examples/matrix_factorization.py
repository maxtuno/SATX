import numpy
import satx

n = 5

M = numpy.random.randint(-10, 10, size=(n, n))

satx.engine(10)

A = satx.matrix(dimensions=(n, n))
B = satx.matrix(dimensions=(n, n))

assert (numpy.matmul(A, B) == M).all()

if satx.satisfy(turbo=True, log=True):
    print(M)
    print(80 * '-')
    A = numpy.vectorize(int)(A)
    B = numpy.vectorize(int)(B)
    print(A)
    print(B)
    print(80 * '-')
    print(numpy.matmul(A, B))
else:
    print('Infeasible...')
