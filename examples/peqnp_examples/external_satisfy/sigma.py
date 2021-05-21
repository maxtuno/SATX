import satx

satx.engine(16, cnf='sigma.cnf')

x = satx.natural()
n = satx.natural()

satx.sigma(lambda k: k ** 2, 1, n) == x

while satx.external_satisfy(solver='java -jar -Xmx4g blue.jar'):
    print(x, n, sum(k ** 2 for k in range(1, n.value + 1)))
