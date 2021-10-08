"""
My son came to me the other day and said, "Dad, I need help with a math problem."
The problem went like this:
- We're going out to dinner taking 1-6 grandparents, 1-10 parents and/or 1-40 children
- Grandparents cost $3 for dinner, parents $2 and children $0.50
- There must be 20 total people at dinner and it must cost $20
How many grandparents, parents and children are going to dinner?
"""

import satx

satx.engine(10, cnf_path='aux.cnf')

# g is the number of grandparents
g = satx.integer()
# c is the number of children
c = satx.integer()
# p is the number of parents
p = satx.integer()

assert 1 <= g <= 7
assert 1 <= p <= 11
assert 1 <= c <= 41

assert g * 6 + p * 2 + c * 1 == 40
assert g + p + c == 20

while satx.satisfy('slime'):
    print(g, p, c)
