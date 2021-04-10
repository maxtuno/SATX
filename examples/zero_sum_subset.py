import random
import satx

size = 100
bits = 16

data = [random.randint(-2 ** bits, 2 ** bits) for _ in range(size)]
while 0 in data:
    data.remove(0)

print(data)

satx.engine(sum(map(abs, data)).bit_length())

x = satx.tensor(dimensions=(len(data),))

assert sum([x[[i]](0, data[i]) for i in range(len(data) // 2)]) == -sum([x[[i]](0, data[i]) for i in range(len(data) // 2, len(data))])

while satx.satisfy():
    sub = [data[i] for i in range(len(data)) if x.binary[i]]
    print(sum(sub), sub)
