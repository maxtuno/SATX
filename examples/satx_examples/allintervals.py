import satx

n = 12

satx.engine(n.bit_length() + 1)

xs = satx.vector(size=n)
ys = [abs(xs[i + 1] - xs[i]) for i in range(n - 1)]

satx.apply_single(xs, lambda x: 0 <= x < n)
satx.apply_single(ys, lambda y: 0 < y <= n)

satx.all_different(xs)
satx.all_different(ys)

count = 0
while satx.satisfy():
    print(xs, ys)
    count += 1

print('TOTAL: {}'.format(count))