import satx

satx.engine(10)

x = satx.integer()
y = satx.integer()

assert x ** 2 + 1 == y

while satx.satisfy():
    if x ** 2 + 1 == y:
        print(x, y, x ** 2 + 1 == y)
    else:
        print(x, satx.oo() + y, x ** 2 + 1 == satx.oo() + y)