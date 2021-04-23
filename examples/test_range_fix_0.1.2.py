import satx

satx.engine(32)

a = satx.integer()

A = satx.one_of([-a, a])

assert -1000 <= A ** 3 <= 1000

while satx.satisfy():
    print(-1000 <= a ** 3 <= 1000, a)