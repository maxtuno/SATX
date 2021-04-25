import satx.gcc

rsa = 897630689

satx.engine(rsa.bit_length() + 1)

p = satx.integer()
q = satx.integer()

satx.gcc.gcd(p, rsa, q)

assert q != 1

if satx.satisfy(turbo=True, log=True):
    print(q, rsa % q.value == 0)
else:
    print('Infeasible...')
