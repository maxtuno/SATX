"""
See model in OscaR

A number with an interesting property: when I divide it by v, the remainder is v-1,
and this from v ranging from 2 to 9.
It's not a small number, but it's not really big, either.
When I looked for a smaller number with this property I couldn't find one.
Can you find it?
"""

import satx

satx.engine(14, cnf_path='aux.cnf')

x = satx.integer()

for i in range(2, 10):
    assert x % i == i - 1

while satx.satisfy('kissat'):
    print(x)
