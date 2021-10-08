"""
Problem 016 on CSPLib
"""

import satx

satx.engine(32, cnf_path='aux.cnf')

mapping = {1: 'r', 2: 'ry', 3: 'g', 4: 'y'}

R, RY, G, Y = 1, 2, 3, 4

table = [(R, R, G, G), (RY, R, Y, R), (G, G, R, R), (Y, R, RY, R)]

# v[i] is the color for the ith vehicle traffic light
v = satx.vector(size=4)
# p[i] is the color for the ith pedestrian traffic light
p = satx.vector(size=4)

satx.apply_single(v, lambda t: t.is_in([R, RY, G, Y]))
satx.apply_single(p, lambda t: t.is_in([R, G]))

for i in range(4):
    assert satx.dot([v[i], p[i], v[(i + 1) % 4], p[(i + 1) % 4]], [1, 10, 100, 1000]) == satx.one_of([satx.dot(t, [1, 10, 100, 1000]) for t in table])

while satx.satisfy('kissat'):
    vv = [mapping[t.value] for t in v]
    pp = [mapping[t.value] for t in p]
    for a, b in zip(vv, pp):
        print(a, b, end=', ')
    print()
    print(80 * '-')
