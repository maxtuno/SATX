import satx.gcc

n = 5

satx.engine(10)

lst = satx.vector(size=5)

satx.apply_single(lst, lambda t: -n <= t <= n)

satx.gcc.all_different(lst)

while satx.satisfy():
    print(lst)
    if len(set(satx.values(lst))) != n:
        raise Exception('ERROR')
