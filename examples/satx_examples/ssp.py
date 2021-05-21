import random
import satx

bits = 10
universe = [random.randint(1, 2 ** bits) for _ in range(100)]
t = random.randint(min(universe), sum(universe))

while 0 in universe:
    universe.remove(0)

print(t, universe)

satx.engine(t.bit_length())

x = satx.tensor(dimensions=(len(universe)))

assert sum(x[[i]](0, universe[i]) for i in range(len(universe))) == t

if satx.satisfy(turbo=True):
    sub = [universe[i] for i in range(len(universe)) if x.binary[i]]
    print(t, sum(sub), sub)
else:
    print('Infeasible ...')
