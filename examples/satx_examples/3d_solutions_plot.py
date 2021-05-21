import satx
import matplotlib.pyplot as plt

satx.engine(16)

x = satx.integer()
y = satx.integer()
z = satx.integer()

assert x ** 3 == y + z ** 5

x = satx.one_of([-x, x])
y = satx.one_of([-y, y])
z = satx.one_of([-z, z])

ss = []
while satx.satisfy():
    if x ** 3 == y + z ** 5:
        print(x, y, z)
        ss.append((x.value, y.value, z.value))
    else:
        # print('ERROR: Currently there are values, which are related to the overflow of the values compiled to CNF, the objective is to solve this problem in future versions.')
        pass

xs, ys, zs = zip(*ss)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xs, ys, zs)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
