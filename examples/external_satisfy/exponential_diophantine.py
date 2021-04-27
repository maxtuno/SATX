import satx

satx.engine(24, cnf='exponential_diophantine.cnf')

_2 = satx.constant(2)
n = satx.integer()
x = satx.integer()

assert _2 ** n - 7 == x ** 2

while satx.external_satisfy(solver='java -jar -Xmx4g blue.jar'):
    print(n, x)
