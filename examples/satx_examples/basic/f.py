import satx

satx.engine(10)

x = satx.integer()
y = satx.integer()

assert -100 <= x <= 100
assert -100 <= y <= 100

assert x ** 2 + y ** 2 == 100

while satx.satisfy():
    print(x, y)
