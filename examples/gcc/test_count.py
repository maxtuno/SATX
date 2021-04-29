import satx.gcc

n = 5

satx.engine(10)

lst = satx.vector(size=n)

satx.gcc.count(5, lst, lambda a, b: a >= b, 3)

while satx.satisfy():
    print(lst)
    if lst.count(5) < 3:
        raise Exception('ERROR')
