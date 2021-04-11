"""
///////////////////////////////////////////////////////////////////////////////
//        Copyright (c) 2012-2021 Oscar Riveros. all rights reserved.        //
//                        oscar.riveros@peqnp.science                        //
//                                                                           //
//   without any restriction, Oscar Riveros reserved rights, patents and     //
//  commercialization of this knowledge or derived directly from this work.  //
///////////////////////////////////////////////////////////////////////////////

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import satx

# This SATX script try to put *all* pieces of full (size) sets in a n x m grid.

if __name__ == "__main__":

    fig, ax = plt.subplots()
    cmap = matplotlib.colors.ListedColormap(['black', 'white'])

    n = 8
    m = 4
    size = 1  # number of full sets

    if n * m < 7 * 4 * size:
        print('Inconsistent...')
        exit(0)

    satx.engine(3)

    aux = []
    for _ in range(size):
        # L
        x = np.asarray(satx.matrix(dimensions=(n, m)))
        aux.append(x)
        satx.all_binaries(x.flatten())
        xs = []
        assert x.sum() == 4
        for i in range(n):
            for j in range(m - 2):
                for k in range(n - i - 1):
                    xs += [x[i + k][j] & x[i + 1 + k][j] & x[i + 1 + k][j + 1] & x[i + 1 + k][j + 2]]
                    xs += [x.T[j][i + k] & x.T[j][i + 1 + k] & x.T[j + 1][i + 1 + k] & x.T[j + 2][i + 1 + k]]
        assert satx.one_of(xs) == 1

        # -L
        x = np.asarray(satx.matrix(dimensions=(n, m)))
        aux.append(x)
        satx.all_binaries(x.flatten())
        xs = []
        assert x.sum() == 4
        for i in range(n):
            for j in range(2, m):
                for k in range(n - i - 1):
                    xs += [x[i + k][j] & x[i + 1 + k][j] & x[i + 1 + k][j - 1] & x[i + 1 + k][j - 2]]
                    xs += [x.T[j][i + k] & x.T[j][i + 1 + k] & x.T[j - 1][i + 1 + k] & x.T[j - 2][i + 1 + k]]
        assert satx.one_of(xs) == 1

        # I
        x = np.asarray(satx.matrix(dimensions=(n, m)))
        aux.append(x)
        satx.all_binaries(x.flatten())
        xs = []
        assert x.sum() == 4
        for i in range(n):
            for j in range(m):
                for k in range(n - i - 3):
                    xs += [x[i + k][j] & x[i + 1 + k][j] & x[i + 2 + k][j] & x[i + 3 + k][j]]
                    xs += [x.T[j][i + k] & x.T[j][i + 1 + k] & x.T[j][i + 2 + k] & x.T[j][i + 3 + k]]
        assert satx.one_of(xs) == 1

        # []
        x = np.asarray(satx.matrix(dimensions=(n, m)))
        aux.append(x)
        satx.all_binaries(x.flatten())
        xs = []
        assert x.sum() == 4
        for i in range(n):
            for j in range(m - 1):
                for k in range(n - i - 1):
                    xs += [x[i + k][j] & x[i + 1 + k][j] & x[i + k][j + 1] & x[i + 1 + k][j + 1]]
                    xs += [x.T[j][i + k] & x.T[j][i + 1 + k] & x.T[j + 1][i + k] & x.T[j + 1][i + 1 + k]]
        assert satx.one_of(xs) == 1

        # S
        x = np.asarray(satx.matrix(dimensions=(n, m)))
        aux.append(x)
        satx.all_binaries(x.flatten())
        xs = []
        assert x.sum() == 4
        for i in range(n):
            for j in range(m - 1):
                for k in range(n - i - 2):
                    xs += [x[i + k][j] & x[i + 1 + k][j] & x[i + 2 + k][j + 1] & x[i + 1 + k][j + 1]]
                    xs += [x.T[j][i + k] & x.T[j][i + 1 + k] & x.T[j + 1][i + 2 + k] & x.T[j + 1][i + 1 + k]]
        assert satx.one_of(xs) == 1

        # -S
        x = np.asarray(satx.matrix(dimensions=(n, m)))
        aux.append(x)
        satx.all_binaries(x.flatten())
        xs = []
        assert x.sum() == 4
        for i in range(n):
            for j in range(1, m):
                for k in range(n - i - 2):
                    xs += [x[i + k][j] & x[i + 1 + k][j] & x[i + 2 + k][j - 1] & x[i + 1 + k][j - 1]]
                    xs += [x.T[j][i + k] & x.T[j][i + 1 + k] & x.T[j - 1][i + 2 + k] & x.T[j - 1][i + 1 + k]]
        assert satx.one_of(xs) == 1

        # T
        x = np.asarray(satx.matrix(dimensions=(n, m)))
        aux.append(x)
        satx.all_binaries(x.flatten())
        xs = []
        assert x.sum() == 4
        for i in range(n):
            for j in range(m - 1):
                for k in range(n - i - 2):
                    xs += [x[i + k][j] & x[i + 1 + k][j] & x[i + 2 + k][j] & x[i + 1 + k][j + 1]]
                    xs += [x.T[j][i + k] & x.T[j][i + 1 + k] & x.T[j][i + 2 + k] & x.T[j + 1][i + 1 + k]]
        assert satx.one_of(xs) == 1

    for i in range(len(aux)):
        for j in range(len(aux)):
            if i < j:
                assert (aux[i] & aux[j]).sum() == 0

    if satx.satisfy(turbo=True, log=True):
        print(sum(map(np.vectorize(int), aux)))
        for i, x in enumerate(aux):
            ax.imshow(np.vectorize(int)(x), interpolation='none', cmap=cmap, alpha=0.2)
            print(np.vectorize(int)(x))
    else:
        print('Infeasible...')

    plt.savefig('tetris_{}_{}_{}.png'.format(n, m, size))
