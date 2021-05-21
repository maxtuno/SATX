import satx

rsa = 3007

satx.engine(rsa.bit_length(), cnf='remote.cnf')

p = satx.natural()
q = satx.natural()

assert p * q == rsa

while satx.external_satisfy(solver='python3 remote_solver.py'):
    print(p, q)
