import multiprocessing

import satx


def parallelize(idx):
    satx.engine(16, cnf='test-{}.cnf'.format(idx))

    x = satx.natural()
    y = satx.natural()
    z = satx.natural()

    assert x ** y - z ** 2 == 666

    if satx.external_satisfy('./slime', params='-massive'):
        print(x, y, z)
        return x.value, y.value, z.value
    else:
        print('Infeasible...')
    satx.external_reset()
    return None


if __name__ == '__main__':
    multiprocessing.freeze_support()

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        print(pool.map(parallelize, range(4)))
