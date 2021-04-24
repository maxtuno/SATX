import satx.gcc

lst = [4, 8, 1, 0, 3, 3, 4, 3]

satx.engine(10)

idx = satx.integer()
val = satx.integer()

satx.gcc.element(idx, lst, val)

while satx.satisfy():
    print(idx, lst, val)
