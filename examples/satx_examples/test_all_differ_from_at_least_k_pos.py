import satx
import satx.gcc

if __name__ == '__main__':
    n = 3
    k = 4

    satx.engine(4)

    vectors = [satx.vector(size=5) for _ in range(n)]

    for V in vectors:
        satx.apply_single(V, lambda t: -5 <= t <= 5)

    satx.gcc.all_differ_from_at_least_k_pos(k, vectors)

    while satx.satisfy():
        for i1 in range(len(vectors) - 1):
            for i2 in range(i1 + 1, len(vectors)):
                s = ''
                for j in range(len(vectors[0])):
                    if vectors[i1][j] == vectors[i2][j]:
                        s += '1'
                    else:
                        s += '0'
                print(s, s.count('0'), s.count('0') >= k)
                if s.count('0') < k:
                    raise Exception('ERROR')
        print(80 * '-')