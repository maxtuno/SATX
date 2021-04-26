# https://www.sat-x.io/2021/04/26/protein-folding

import satx.gcc

acids = list(range(1, 51))  # list of amino acids and hydrophobic acids

h_phobic = [2, 4, 5, 6, 11, 12, 17, 20, 21, 25, 27, 28, 30, 31, 33, 37, 44, 46]

ij = []  # Creating the data structures to generate the model
# Indices of hydrophobic acids that can be matched
for i in h_phobic:
    for j in h_phobic:
        if j > i + 1:
            tp = i, j
            ij.append(tp)

ik1j = []  # Indices for constraints of type 1
ik2j = []  # Indices for constraints of type 2
for i, j in ij:
    for k in range(i, j):
        if k == (i + j - 1) / 2:
            tp = i, j, k
            ik2j.append(tp)
        else:
            tp = i, j, k
            ik1j.append(tp)

ij_fold = []  # Matching that are enabled by a folding
for i, j, k in ik2j:
    tp = i, j
    ij_fold.append(tp)

opt = 0
while True:
    satx.engine(len(acids).bit_length())

    match = satx.tensor(dimensions=(len(acids), len(acids)))  # Matching variables
    fold = satx.tensor(dimensions=(len(acids),))  # Folding variables

    for i, j, k in ik1j:
        assert fold[[k]](0, 1) + match[[i, j]](0, 1) <= 1

    for i, j, k in ik2j:
        assert match[[i, j]](0, 1) <= fold[[k]](0, 1)

    assert sum(match[[i, j]](0, 1) for i, j in ij_fold) > opt

    if satx.satisfy(turbo=True):
        opt = sum(match.binary[i][j] for i, j in ij_fold)
        # Output report
        print('Optimal number of hydrophobic acids matching: {}'.format(opt))
        print('_______________________________________')
        print('Optimal matching of hydrophobic acids. ')
        print('_______________________________________')
        for i, j, k in ik2j:
            if match.binary[i][j]:
                print('Hydrophobic acid matching {}, {} with folding at amon acid {}.'.format(i, j, k))
        print('=======================================')
    else:
        print('OPTIMAL FOUND.')
        break
