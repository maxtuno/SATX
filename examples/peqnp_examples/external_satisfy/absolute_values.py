import satx

satx.engine(4, cnf='absolute_values.cnf')

x = satx.integer()
y = satx.integer()

satx.apply_single([x, y], lambda t: -7 < t < 7)

assert abs(x - y) == 1

while satx.external_satisfy(solver='java -jar -Xmx4g blue.jar'):
    print(x, y, x - y, abs(x - y))