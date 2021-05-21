import satx

data = [2, 3, 5, 7, 1]

satx.engine(10)

xs, ys = satx.subsets(data, complement=True)

assert sum(xs) == sum(ys)

while satx.satisfy():
    print(xs, ys)