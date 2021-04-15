import satx

data = [2, 3, 5, 7, 1]

satx.engine(10)

xs, ys = satx.permutations(data, n=3)

while satx.satisfy():
    print(xs, ys)