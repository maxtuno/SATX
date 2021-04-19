import numpy
import satx

# ref https://www.sat-x.io/2021/04/19/three-dimensional-noughts-and-crosses/

'''
We first create a list of all possible lines and diagonals in a 3-D tic-tac-toe board. 
Each is represented as a Python tuple with 3 entries, where each entry gives the (i,j,k) 
position of the corresponding cell. There are 49 in total.
'''

n = 3

lines = []
for i in range(n):
    for j in range(n):
        for k in range(n):
            if i == 0:
                lines.append(((0, j, k), (1, j, k), (2, j, k)))
            if j == 0:
                lines.append(((i, 0, k), (i, 1, k), (i, 2, k)))
            if k == 0:
                lines.append(((i, j, 0), (i, j, 1), (i, j, 2)))
            if i == 0 and j == 0:
                lines.append(((0, 0, k), (1, 1, k), (2, 2, k)))
            if i == 0 and j == 2:
                lines.append(((0, 2, k), (1, 1, k), (2, 0, k)))
            if i == 0 and k == 0:
                lines.append(((0, j, 0), (1, j, 1), (2, j, 2)))
            if i == 0 and k == 2:
                lines.append(((0, j, 2), (1, j, 1), (2, j, 0)))
            if j == 0 and k == 0:
                lines.append(((i, 0, 0), (i, 1, 1), (i, 2, 2)))
            if j == 0 and k == 2:
                lines.append(((i, 0, 2), (i, 1, 1), (i, 2, 0)))

lines.append(((0, 0, 0), (1, 1, 1), (2, 2, 2)))
lines.append(((2, 0, 0), (1, 1, 1), (0, 2, 2)))
lines.append(((0, 2, 0), (1, 1, 1), (2, 0, 2)))
lines.append(((0, 0, 2), (1, 1, 1), (2, 2, 0)))

'''
So if we minimize the sum total of the (absolute) differences between the amount 
of black and withe balls in each line, we will obtain our solution. 
'''

opt = 59  # This value can be anything between, the range defined by the engine bits.
while True:
    satx.engine(10)  # only 10 bits are needed to solve the problem.

    B = satx.tensor(dimensions=(n, n, n))  # define a tensor of n x n x n, for a basic introduction to tensor see http://www.peqnp.com.

    C = satx.constant(0)  # we define a simple accumulator were store the differences of black and withe balls.
    for line in lines:
        X = B[line[0]](0, 1) + B[line[1]](0, 1) + B[line[2]](0, 1)  # count blacks on current line.
        O = B[line[0]](1, 0) + B[line[1]](1, 0) + B[line[2]](1, 0)  # count withes on current line.
        C += abs(X - O)  # accumulate the current difference.

    assert C < opt  # ensure that this differece is lower that the current optimal.

    if satx.satisfy(turbo=True):
        opt = C.value  # assign the current optimal to global optimal for the next loop
        print(opt)  # print the results
        print(numpy.vectorize(int)(B.binary))  # use numpy only for a beatty output.
        print(80 * '-')  # separate solutions.
    else:
        # if not more sub optimal values then is optimal and exit.
        break
