import satx

satx.engine(10)

x = satx.integer()
y = satx.integer()

assert -10 <= x < 10
assert -10 < y <= 10

assert x != y

while satx.satisfy():
    print(x, y)
