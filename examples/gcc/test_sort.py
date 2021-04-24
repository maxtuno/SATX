import satx.gcc

n = 5

satx.engine(10)

lst1 = satx.vector(size=n)
lst2 = satx.vector(size=n)

satx.apply_single(lst1, lambda t: -n <= t <= n)

satx.gcc.sort(lst1, lst2)

while satx.satisfy():
    print(lst1, lst2)
    if sorted(satx.values(lst1)) != satx.values(lst2):
        raise Exception('ERROR')
