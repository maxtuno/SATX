import functools
import multiprocessing

import numpy
import satx


def distance(seq, ds):
    return sum([round(numpy.linalg.norm(ds[seq[i - 1]] - ds[seq[i]])) for i in range(len(ds))])


def parallelize(seed, ds):
    n = len(ds)
    numpy.random.seed(seed)
    seq = numpy.arange(n)
    numpy.random.shuffle(seq)
    seq = satx.hess_sequence(n, oracle=functools.partial(distance, ds=ds), seq=seq)
    print(distance(seq, ds), seq)
    return distance(seq, ds), seq


if __name__ == '__main__':
    with open('xqf131.txt', 'r') as file:
        lines = file.readlines()
        data = int(lines[0]) * [None]
        del lines[0]
        for line in lines:
            idx, x, y = map(int, line.split(' '))
            data[idx - 1] = numpy.asarray([x, y])

    satx.engine()

    multiprocessing.freeze_support()

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        print(min(pool.map(functools.partial(parallelize, ds=data), numpy.random.randint(0, 2 ** 32, size=8))))
