import satx

rsa = 3007

satx.engine(rsa.bit_length() + 1)

x = satx.integer()
y = satx.integer()

assert x * y == rsa
assert 1 < x < y

while satx.satisfy():
    print(x, y)