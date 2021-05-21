import satx

satx.engine(10)

x = satx.integer()

assert -10 <= x < 10

while satx.satisfy():
    print(x)
