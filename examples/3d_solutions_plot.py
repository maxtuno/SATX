import satx
import matplotlib.pyplot as plt

satx.engine(16)

x = satx.integer()
y = satx.integer()
z = satx.integer()

assert x ** 3 == y + z ** 5

ss = []
while satx.satisfy():
    print(x, y, z)
    ss.append((x.value, y.value, z.value))

xs, ys, zs = zip(*ss)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xs, ys, zs)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
