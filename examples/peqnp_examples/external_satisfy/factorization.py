import satx

rsa = 3007

satx.engine(rsa.bit_length(), cnf='factorization.cnf')

p = satx.natural()
q = satx.natural()
assert p * q == rsa
assert 1 < p < q

while satx.external_satisfy(solver='java -jar -Xmx4g blue.jar'):
    print(p, q)