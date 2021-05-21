import satx.gcc

satx.engine(13)

lst_from = [1, 9, 1, 5, 2, 1]
lst_to = [1, 1, 1, 2, 5, 9]
lst_per = satx.vector(size=len(lst_to))

satx.gcc.sort_permutation(lst_from, lst_per, lst_to)

while satx.satisfy():
    print(lst_per, [lst_from[satx.values(lst_per)[i]] for i in range(len(lst_to))], lst_to)
