## SAT-X [![Downloads](https://pepy.tech/badge/satx)](https://pepy.tech/project/satx) 
### The constraint modeling language for SAT solvers

SAT-X is a language for constrained optimization and decision problems over positive integers, that work with any SAT Competition standard SAT solver. Is based on Python, and is ase to learn and easy to use with all technologies associated to this language.

# Some excelent SAT Solvers

1- https://github.com/maxtuno/SLIME (standalone and cloud [MPI] - Oscar Riveros)

2- https://github.com/arminbiere/kissat (The Kissat SAT Solver - Armin Biere)

3- https://github.com/maxtuno/MiniSat (MiniSat 2.2.0 with DRUP proof, CMake and StarExec Ready. -  Niklas Een, Niklas Sorensson)

4- https://github.com/maxtuno/blue (A Powerful SAT Solver for Java - Oscar Riveros)

5- Any SAT Solver with the SAT Competition standars. (http://www.satcompetition.org)


## Installation
```python
pip install satx
```

### Semi Magic Square of Squares

```python
import satx

satx.engine(bits=22, cnf_path='tmp.cnf')

p = satx.integer()
q = satx.integer()
r = satx.integer()
s = satx.integer()

satx.apply_single([p, q, r, s], lambda x: x > 0)

A = (p ** 2 + q ** 2 - r ** 2 - s ** 2) ** 2
B = (2 * (q * r + p * s)) ** 2
C = (2 * (p * r - q * s)) ** 2

D = (2 * (q * r - p * s)) ** 2
E = (p ** 2 - q ** 2 + r ** 2 - s ** 2) ** 2
F = (2 * (r * s + p * q)) ** 2

G = (2 * (q * s + p * r)) ** 2
H = (2 * (p * q - r * s)) ** 2
I = (p ** 2 - q ** 2 - r ** 2 + s ** 2) ** 2

assert E + I == B + C
# assert G + E == F + I # perfect magic

if satx.satisfy('slime'):
    print(p, q, r, s)
    print(80 * '-')

    print([A, B, C])
    print([D, E, F])
    print([G, H, I])

    print(80 * '-')

    print(sum([A, B, C]))
    print(sum([D, E, F]))
    print(sum([G, H, I]))

    print(80 * '-')

    print(sum([A, D, G]))
    print(sum([B, E, H]))
    print(sum([C, F, I]))

    print(80 * '-')

    print(sum([A, E, I]))
    print(sum([C, E, G]))
else:
    print('Infeasible ... bits={}'.format(satx.bits()))
```

```
38 21 16 5
--------------------------------------------------------------------------------
[2572816, 1106704, 1012036]
[85264, 1522756, 3083536]
[2033476, 2062096, 595984]
--------------------------------------------------------------------------------
4691556
4691556
4691556
--------------------------------------------------------------------------------
4691556
4691556
4691556
--------------------------------------------------------------------------------
4691556
4568268
```

```
[1604**2, 1052**2, 1006**2]
[292**2, 1234**2, 1756**2]
[1426**2, 1436**2, 772**2]
```

Note: Documentation is on development, for a general reference and examples see www.peqnp.com that is an old version of SAT-X but work with precompiled SLIME sat solver.

you are welcome to contribute to the proyect, with examples, documentation, courses, slides, improvements... etc.

<img src="https://cr-ss-service.azurewebsites.net/api/ScreenShot?widget=summary&username=maxtuno&badges=2&show-avatar=true&style=--header-bg-color:%23000;--border-radius:10px"/>
