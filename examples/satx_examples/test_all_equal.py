import satx.gcc

n = 10

satx.engine(10)

vars = satx.vector(size=n)

satx.gcc.all_equal(vars)

while satx.satisfy():
    print(vars)
