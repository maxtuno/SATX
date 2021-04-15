import satx

data = [2, 3, 5, 7, 1]

satx.engine(10)

xs, ys = satx.subset(data, k=3, empty=-1, complement=True)

while satx.satisfy():
    print(xs, ys)