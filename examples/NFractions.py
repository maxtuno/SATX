"""
Problem 041 on CSPLib
"""

import satx

satx.engine(16, cnf_path='aux.cnf')

digits = satx.vector(size=9)

satx.apply_single(digits, lambda t: 0 < t < 10)

a, b, c, d, e, f, g, h, i = digits

satx.all_different(digits)

assert a * (10 * e + f) * (10 * h + i) + d * (10 * b + c) * (10 * h + i) + g * (10 * b + c) * (10 * e * f) == (10 * b + c) * (10 * e + f) * (10 * h + i)

while satx.satisfy('kissat'):
    print(a, b, c, d, e, f, g, h, i)
