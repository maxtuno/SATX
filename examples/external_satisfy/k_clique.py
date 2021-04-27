import sys

import satx


def load_file(file_name):
    graph_ = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            if line.startswith('p edge '):
                n_ = int(line[len('p edge '):].split(' ')[0])
            elif line.startswith('e '):
                x, y = map(int, line[len('e '):].split(' '))
                graph_.append((x, y))
    matrix_ = [[0 for _ in range(n_)] for _ in range(n_)]
    for i, j in graph_:
        matrix_[i - 1][j - 1] = 1
        matrix_[j - 1][i - 1] = 1
    return n_, matrix_


if __name__ == '__main__':

    # python3 k_clique.py graph.txt 10 > graph.sol
    # ./k_clique_validator graph.txt graph.sol

    n, adj = load_file(sys.argv[1])
    k = int(sys.argv[2])

    satx.engine(k.bit_length() + 1, cnf='k_clique.cnf')
    bits = satx.tensor(dimensions=(n,))
    assert sum(bits[[i]](0, 1) for i in range(n)) == k
    for i in range(n - 1):
        for j in range(i + 1, n):
            if adj[i][j] == 0:
                assert bits[[i]](0, 1) & bits[[j]](0, 1) == 0

    if satx.external_satisfy(solver='slime'):
        print(k)
        print(' '.join([str(i + 1) for i in range(n) if bits.binary[i]]))
    else:
        print('Infeasible ...')
