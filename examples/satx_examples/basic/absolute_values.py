import satx

satx.engine(10)

x = satx.integer()
y = satx.integer()

assert abs(x - y) == 1

while satx.satisfy():
    print(x, y, x - y, abs(x - y))