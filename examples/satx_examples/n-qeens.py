import satx

# https://www.sat-x.io/2021/04/15/n-queens

n = 8

satx.engine(n.bit_length() + 1)

qs = satx.vector(size=n)

satx.all_different(qs)
satx.all_different([qs[i] + i for i in range(n)])
satx.all_different([qs[i] - i for i in range(n)])

satx.apply_single(qs, lambda x: 0 <= x < n)

count = 0
while satx.satisfy():
    print(qs)
    for i in range(n):
        print(''.join(['Q ' if qs[i] == j else '. ' for j in range(n)]))
    print('')
    count += 1

print('TOTAL: {}'.format(count))

