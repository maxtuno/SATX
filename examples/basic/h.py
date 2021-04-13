import satx

satx.engine(10)

x = satx.integer()
y = satx.integer()

assert -100 <= x <= 100
assert -100 <= y <= 100

assert x * y == 100

while satx.satisfy():
    print(x, y)
