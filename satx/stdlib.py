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

"""
The standard high level library for the SAT-X system.
"""

from .gaussian import Gaussian
from .linear import Linear
from .rational import Rational
from .alu import *

csp = None


def version():
    """
    Print the information about the system.
    """
    print('SAT-X Mathematical Solver from http://www.peqnp.com')
    print('Copyright (c) 2012-2021 Oscar Riveros. all rights reserved.')
    print('oscar.riveros@peqnp.science')


def check_engine():
    if csp is None:
        print('The SAT-X system is not initialized.')
        exit(0)


def engine(bits=None, info=False):
    """
    Initialize or reset the SAT-X system.
    :param bits: Implies an $[-2^{bits}, 2^{bits})$ search space.
    :param info: Print the information about the system.
    """
    global csp
    csp = ALU(0 if not bits else bits)
    if info:
        version()


def slime_cli(cnf_path, model_path='', proof_path=''):
    """
    Use the SLIME SAT solver directly.
    :param cnf_path: The CNF file to solve.
    :param model_path: The path to the model if SAT (optional).
    :param proof_path: The path for the DRUP-PROOF if UNSAT (optional).
    :return: A list with the model if the instance is SAT or an empty list if it is UNSAT.
    """
    import slime
    return slime.slime_cli(cnf_path, model_path, proof_path)


def integer(bits=None):
    """
    Correspond to an integer.
    :param bits: The bits for the integer.
    :return: An instance of Integer.
    """
    global csp
    check_engine()
    csp.variables.append(csp.int(size=bits))
    return csp.variables[-1]


def constant(value, bits=None):
    """
    Correspond to an constant.
    :param bits: The bits for the constant.
    :param value: The value of the constant.
    :return: An instance of Constant.
    """
    global csp
    check_engine()
    csp.variables.append(csp.int(size=bits, value=value))
    return csp.variables[-1]


def satisfy(turbo=False, log=False):
    """
    Find a model for the current problem.
    :param turbo: This make a simplification of the model, is more fast to solve, but destroy the internal structure of the problem, need regenerate, use for only one solution.
    :param log: Show log for the SLIME SAT Solver.
    :return: True if SATISFIABLE else False
    """
    return csp.to_sat(solve=True, turbo=turbo, log=log, assumptions=None, cnf_path='', model_path='', proof_path='')


def subsets(lst, k=None, complement=False):
    """
    Generate all subsets for an specific universe of data.
    :param lst: The universe of data.
    :param k: The cardinality of the subsets.
    :param complement: True if include the complement in return .
    :return: (binary representation of subsets, the generic subset representation, the complement of subset if complement=True)
    """
    global csp
    check_engine()
    bits = csp.int(size=len(lst))
    csp.variables.append(bits)
    if k is not None:
        assert sum(csp.zero.iff(-bits[i], csp.one) for i in range(len(lst))) == k
    subset_ = [csp.zero.iff(-bits[i], lst[i]) for i in range(len(lst))]
    csp.variables += subset_
    if complement:
        complement_ = [csp.zero.iff(bits[i], lst[i]) for i in range(len(lst))]
        csp.variables += complement_
        return subset_, complement_
    else:
        return subset_


def subset(lst, k, empty=None, complement=False):
    """
    An operative structure (like integer ot constant) that represent a subset of at most k elements.
    :param lst: The data for the subsets.
    :param k: The maximal bits for subsets.
    :param empty: The empty element, 0, by default.
    :param complement: True if include in return the complement.
    :return: An instance of subset or (subset, complement) if complement=True.
    """
    global csp
    check_engine()
    if complement:
        subset_, complement_ = csp.subset(k, lst, empty, complement=complement)
    else:
        subset_ = csp.subset(k, lst, empty)
    csp.variables += subset_
    if complement:
        csp.variables += complement_
        return subset_, complement_
    return subset_


def vector(bits=None, size=None, is_gaussian=False, is_rational=False, is_mip=False, is_real=False):
    """
    A vector of integers.
    :param bits: The bit bits for each integer.
    :param size: The bits of the vector.
    :param is_gaussian: Indicate of is a Gaussian Integers vector.
    :param is_rational: Indicate of is a Rational vector.
    :param is_mip: Indicate of is a MIP vector.
    :param is_real: Indicate of is a MIP vector and is real or int.
    :return: An instance of vector.
    """
    global csp
    check_engine()
    if is_rational:
        return [rational() for _ in range(size)]
    if is_gaussian:
        return [gaussian() for _ in range(size)]
    if is_mip:
        lns = []
        for _ in range(size):
            lns.append(linear(is_real=is_real))
        return lns
    else:
        array_ = csp.array(size=bits, dimension=size)
        csp.variables += array_
    return array_


def matrix(bits=None, dimensions=None, is_gaussian=False, is_rational=False, is_mip=False, is_real=False):
    """
    A matrix of integers.
    :param bits: The bit bits for each integer.
    :param dimensions: An tuple with the dimensions for the Matrix (n, m).
    :param is_gaussian: Indicate of is a Gaussian Integers vector.
    :param is_rational: Indicate of is a Rational Matrix.
    :param is_mip: Indicate of is a MIP Matrix.
    :param is_real: Indicate of is a MIP Matrix and is real or int.
    :return: An instance of Matrix.
    """
    global csp
    check_engine()
    matrix_ = []
    for i in range(dimensions[0]):
        row = []
        lns = []
        for j in range(dimensions[1]):
            if is_mip:
                lns.append(linear(is_real=is_real))
                row.append(lns[-1])
            else:
                if is_rational:
                    x = integer(bits=bits)
                    y = integer(bits=bits)
                    csp.variables.append(rational(x, y))
                elif is_gaussian:
                    x = integer(bits=bits)
                    y = integer(bits=bits)
                    csp.variables.append(gaussian(x, y))
                else:
                    csp.variables.append(integer(bits=bits))
                row.append(csp.variables[-1])
        matrix_.append(row)
    return matrix_


def matrix_permutation(lst, n):
    """
    This generate the permutations for an square matrix.
    :param lst: The flattened matrix of data, i.e. a vector.
    :param n: The dimension for the square nxn-matrix.
    :return: An tuple with (index for the elements, the elements that represent the indexes)
    """
    global csp
    check_engine()
    xs = vector(size=n)
    ys = vector(size=n)
    csp.apply(xs, single=lambda x: 0 <= x < n)
    csp.apply(xs, dual=lambda a, b: a != b)
    csp.indexing(xs, ys, lst)
    return xs, ys


def permutations(lst, n):
    """
    Entangle all permutations of size n of a list.
    :param lst: The list to entangle.
    :param n: The bits of entanglement.
    :return: (indexes, values)
    """
    check_engine()
    xs = vector(size=n)
    ys = vector(size=n)
    for i in range(n):
        assert element(ys[i], lst) == xs[i]
    apply_single(xs, lambda a: 0 <= a < n)
    apply_dual(xs, lambda a, b: a != b)
    return xs, ys


def combinations(lst, n):
    """
    Entangle all combinations of bits n for the vector lst.
    :param lst: The list to entangle.
    :param n: The bits of entanglement.
    :return: (indexes, values)
    """
    check_engine()
    xs = vector(size=n)
    ys = vector(size=n)
    for i in range(n):
        assert element(ys[i], lst) == xs[i]
    return xs, ys


def all_binaries(lst):
    """
    This say that, the vector of integer are all binaries.
    :param lst: The vector of integers.
    :return:
    """
    check_engine()
    global csp
    csp.apply(lst, single=lambda arg: 0 <= arg <= 1)


def switch(x, ith, neg=False):
    """
    This conditionally flip the internal bit for an integer.
    :param x: The integer.
    :param ith: Indicate the ith bit.
    :param neg: indicate if the condition is inverted.
    :return: 0 if the uth bit for the argument collapse to true else return 1, if neg is active exchange 1 by 0.
    """
    global csp
    check_engine()
    return csp.zero.iff(-x[ith] if neg else x[ith], csp.one)


def one_of(lst):
    """
    This indicate that at least one of the instruction on the array is active for the current problem.
    :param lst: A list of instructions.
    :return: The entangled structure.
    """
    global csp
    check_engine()
    bits = csp.int(size=len(lst))
    assert sum(bits[[i]](csp.zero, csp.one) for i in range(len(lst))) == csp.one
    return sum(bits[[i]](csp.zero, lst[i]) for i in range(len(lst)))


def factorial(x):
    """
    The factorial for the integer.
    :param x: The integer.
    :return: The factorial.
    """
    global csp
    check_engine()
    return csp.factorial(x)


def sigma(f, i, n):
    """
    The Sum for i to n, for the lambda f f,
    :param f: A lambda f with an standard int parameter.
    :param i: The start for the Sum, an standard int.
    :param n: The integer that represent the end of the Sum.
    :return: The entangled structure.
    """
    global csp
    check_engine()
    return csp.sigma(f, i, n)


def pi(f, i, n):
    """
    The Pi for i to n, for the lambda f f,
    :param f: A lambda f with an standard int parameter.
    :param i: The start for the Pi, an standard int.
    :param n: The integer that represent the end of the Pi.
    :return: The entangled structure.
    """
    global csp
    check_engine()
    return csp.pi(f, i, n)


def dot(xs, ys):
    """
    The dot product of two compatible Vectors.
    :param xs: The fist vector.
    :param ys: The second vector.
    :return: The dot product.
    """
    global csp
    check_engine()
    return csp.dot(xs, ys)


def mul(xs, ys):
    """
    The elementwise product of two Vectors.
    :param xs: The fist vector.
    :param ys: The second vector.
    :return: The product.
    """
    global csp
    check_engine()
    return csp.mul(xs, ys)


def values(lst, cleaner=None):
    """
    Convert to standard values
    :param lst: List with elements.
    :param cleaner: Filter for elements.
    :return: Standard (filtered) values.
    """
    global csp
    check_engine()
    return csp.values(lst, cleaner)


def apply_single(lst, f, indexed=False):
    """
    A sequential operation over a vector.
    :param lst: The vector.
    :param f: The lambda f of one integer variable.
    :param indexed: The lambda f of two integer variable, the first is an index.
    :return: The entangled structure.
    """
    global csp
    check_engine()
    if indexed:
        csp.apply_indexed(lst, single=f)
    else:
        csp.apply(lst, single=f)


def apply_dual(lst, f, indexed=False):
    """
    A cross operation over a vector on all pairs i, j such that i < j elements.
    :param lst: The vector.
    :param f: The lambda f of two integer variables.
    :param indexed: The lambda f of four integer variable, the 2 firsts are indexes.
    :return: The entangled structure.
    """
    global csp
    check_engine()
    if indexed:
        csp.apply_indexed(lst, dual=f)
    else:
        csp.apply(lst, dual=f)


def apply_different(lst, f, indexed=False):
    """
    A cross operation over a vector on all pairs i, j such that i != j elements.
    :param lst: The vector.
    :param f: The lambda f of two integer variables.
    :param indexed: The lambda f of four integer variable, the 2 firsts are indexes.
    :return: The entangled structure.
    """
    global csp
    check_engine()
    if indexed:
        csp.apply_indexed(lst, different=f)
    else:
        csp.apply(lst, different=f)


def all_different(args):
    """
    The all different global constraint.
    :param args: A vector of integers.
    :return:
    """
    global csp
    check_engine()
    csp.apply(args, dual=lambda x, y: x != y)


def all_out(args, values):
    """
    The all different to values global constraint.
    :param args: A vector of integers.
    :param values: The values excluded.
    :return:
    """
    global csp
    check_engine()
    csp.apply(args, single=lambda x: [x != v for v in values])


def all_in(args, values):
    """
    The all in values global constraint.
    :param args: A vector of integers.
    :param values: The values included.
    :return:
    """
    global csp
    check_engine()
    csp.apply(args, single=lambda x: x == one_of(values))


def flatten(mtx):
    """
    Flatten a matrix into list.
    :param mtx: The matrix.
    :return: The entangled structure.
    """
    global csp
    check_engine()
    return csp.flatten(mtx)


def bits():
    """
    The current bits for the engine.
    :return: The bits
    """
    check_engine()
    return csp.bits


def oo():
    """
    The infinite for rhe system, the maximal value for the current engine.
    :return: 2 ** bits - 1
    """
    global csp
    check_engine()
    return csp.oo


def element(item, data):
    """
    Ensure that the element i is on the data, on the position index.
    :param item: The element
    :param data: The data
    :return: The position of element
    """
    global csp
    check_engine()
    ith = integer()
    csp.element(ith, data, item)
    csp.variables.append(ith)
    return csp.variables[-1]


def index(ith, data):
    """
    Ensure that the element i is on the data, on the position index.
    :param ith: The element
    :param data: The data
    :return: The position of element
    """
    global csp
    check_engine()
    item = integer()
    csp.element(ith, data, item)
    csp.variables.append(item)
    return csp.variables[-1]


def gaussian(x=None, y=None):
    """
    Create a gaussian integer from (x+yj).
    :param x: real
    :param y: imaginary
    :return: (x+yj)
    """
    check_engine()
    if x is None and y is None:
        return Gaussian(integer(), integer())
    return Gaussian(x, y)


def rational(x=None, y=None):
    """
    Create a rational x / y.
    :param x: numerator
    :param y: denominator
    :return: x / y
    """
    check_engine()
    if x is None and y is None:
        return Rational(integer(), integer())
    return Rational(x, y)


def at_most_k(x, k):
    """
    At most k bits can be activated for this integer.
    :param x: An integer.
    :param k: k elements
    :return: The encoded variable
    """
    global csp
    check_engine()
    return csp.at_most_k(x, k)


def sqrt(x):
    """
    Define x as a perfect square.
    :param x: The integer
    :return: The square of this integer.
    """
    global csp
    check_engine()
    return csp.sqrt(x)


# ///////////////////////////////////////////////////////////////////////////////
# //        Copyright (c) 2012-2020 Oscar Riveros. all rights reserved.        //
# //                        oscar.riveros@peqnp.science                        //
# //                                                                           //
# //   without any restriction, Oscar Riveros reserved rights, patents and     //
# //  commercialization of this knowledge or derived directly from this work.  //
# ///////////////////////////////////////////////////////////////////////////////
def hess_sequence(n, oracle, fast=False, cycles=1, target=0):
    """
    HESS Algorithm is a Universal Black Box Optimizer (sequence version).
    :param n: The size of sequence.
    :param oracle: The oracle, this output a number and input a sequence.
    :param fast: More fast less accuracy.
    :param cycles: How many times the HESS algorithm is executed.
    :param target: Any value less than this terminates the execution.
    :return optimized sequence.
    """
    xs = list(range(n))
    glb = oracle(xs) + 1
    opt = xs[:]

    def __inv(a, b, xs):
        i, j = min(a, b), max(a, b)
        while i < j:
            xs[i], xs[j] = xs[j], xs[i]
            i += 1
            j -= 1

    top = glb
    for _ in range(cycles):
        glb = top + 1
        if fast:
            while True:
                anchor = glb
                for i in range(len(xs) - 1):
                    for j in range(i + 1, len(xs)):
                        __inv(i, j, xs)
                        loc = oracle(xs)
                        if loc < glb:
                            glb = loc
                            if glb < top:
                                top = glb
                                opt = xs[:]
                                if top <= target:
                                    return opt
                        elif loc > glb:
                            __inv(i, j, xs)
                if anchor == glb:
                    break
        else:
            while True:
                anchor = glb
                for i in range(len(xs)):
                    for j in range(len(xs)):
                        __inv(i, j, xs)
                        loc = oracle(xs)
                        if loc < glb:
                            glb = loc
                            if glb < top:
                                top = glb
                                opt = xs[:]
                                if top <= target:
                                    return opt
                        elif loc > glb:
                            __inv(i, j, xs)
                if anchor == glb:
                    break

    return opt


# ///////////////////////////////////////////////////////////////////////////////
# //        Copyright (c) 2012-2020 Oscar Riveros. all rights reserved.        //
# //                        oscar.riveros@peqnp.science                        //
# //                                                                           //
# //   without any restriction, Oscar Riveros reserved rights, patents and     //
# //  commercialization of this knowledge or derived directly from this work.  //
# ///////////////////////////////////////////////////////////////////////////////
def hess_binary(n, oracle, fast=False, cycles=1, target=0):
    """
    HESS Algorithm is a Universal Black Box Optimizer (binary version).
    :param n: The size of bit vector.
    :param oracle: The oracle, this output a number and input a bit vector.
    :param fast: More fast some times less accuracy.
    :param cycles: How many times the HESS algorithm is executed.
    :param target: Any value less than this terminates the execution.
    :return optimized sequence.
    """
    xs = [False] * n
    glb = oracle(xs) + 1
    opt = xs[:]

    def __inv(i, j, xs):
        if xs[i] == xs[j]:
            xs[i] = not xs[j]
        else:
            aux = xs[i]
            xs[i] = not xs[j]
            xs[j] = aux

    top = glb
    for _ in range(cycles):
        glb = top + 1
        if fast:
            while True:
                anchor = glb
                for i in range(len(xs) - 1):
                    for j in range(i + 1, len(xs)):
                        __inv(i, j, xs)
                        loc = oracle(xs)
                        if loc < glb:
                            glb = loc
                            if glb < top:
                                top = glb
                                opt = xs[:]
                                if top <= target:
                                    return opt
                        elif loc > glb:
                            __inv(i, j, xs)
                if anchor == glb:
                    break
        else:
            while True:
                anchor = glb
                for i in range(len(xs)):
                    for j in range(len(xs)):
                        __inv(i, j, xs)
                        loc = oracle(xs)
                        if loc < glb:
                            glb = loc
                            if glb < top:
                                top = glb
                                opt = xs[:]
                                if top <= target:
                                    return opt
                        elif loc > glb:
                            __inv(i, j, xs)
                if anchor == glb:
                    break
    return opt


# ///////////////////////////////////////////////////////////////////////////////
# //        Copyright (c) 2012-2020 Oscar Riveros. all rights reserved.        //
# //                        oscar.riveros@peqnp.science                        //
# //                                                                           //
# //   without any restriction, Oscar Riveros reserved rights, patents and     //
# //  commercialization of this knowledge or derived directly from this work.  //
# ///////////////////////////////////////////////////////////////////////////////
def hess_abstract(xs, oracle, f, g, log=None, fast=False, cycles=1, target=0):
    """
    HESS Algorithm is a Universal Black Box Optimizer (abstract version).
    :param xs: The initial vector.
    :param oracle: The oracle, this output a number and input a vector.
    :param f: The f(i, j, xs).
    :param g: The g(i, j, xs).
    :param log: The log(top, opt).
    :param fast: More fast some times less accuracy.
    :param cycles: How many times the HESS algorithm is executed.
    :param target: Any value less than this terminates the execution.
    :return optimized vector over the oracle over and f, g.
    """
    glb = oracle(xs) + 1
    opt = xs[:]

    top = glb
    for _ in range(cycles):
        glb = top + 1
        if fast:
            while True:
                anchor = glb
                for i in range(len(xs) - 1):
                    for j in range(i + 1, len(xs)):
                        f(i, j, xs)
                        loc = oracle(xs)
                        if loc < glb:
                            glb = loc
                            if glb < top:
                                top = glb
                                opt = xs[:]
                                if log is not None:
                                    log(top, opt)
                                if top <= target:
                                    return opt
                        elif loc > glb:
                            g(i, j, xs)
                if anchor == glb:
                    break
        else:
            while True:
                anchor = glb
                for i in range(len(xs)):
                    for j in range(len(xs)):
                        f(i, j, xs)
                        loc = oracle(xs)
                        if loc < glb:
                            glb = loc
                            if glb < top:
                                top = glb
                                opt = xs[:]
                                if log is not None:
                                    log(top, opt)
                                if top <= target:
                                    return opt
                        elif loc > glb:
                            g(i, j, xs)
                if anchor == glb:
                    break
    return opt


# ///////////////////////////////////////////////////////////////////////////////
# //        Copyright (c) 2012-2020 Oscar Riveros. all rights reserved.        //
# //                        oscar.riveros@peqnp.science                        //
# //                                                                           //
# //   without any restriction, Oscar Riveros reserved rights, patents and     //
# //  commercialization of this knowledge or derived directly from this work.  //
# ///////////////////////////////////////////////////////////////////////////////
def hyper_loop(n, m):
    """
    An nested for loop
    :param n: The size of the samples
    :param m: The numbers in the sample 0..m
    :return:
    """
    idx = []
    for k in range(m ** n):
        for _ in range(n):
            idx.append(k % m)
            k //= m
            if len(idx) == n:
                yield idx[::-1]
                del idx[:]


def reshape(lst, dimensions):
    """
    Reshape a list
    :param lst: The coherent list to reshape
    :param dimensions:  The list of dimensions
    :return: The reshaped list
    """
    global csp
    check_engine()
    return csp.reshape(lst, dimensions)


def tensor(dimensions):
    """
    Create a tensor
    :param dimensions: The list of dimensions
    :return: A tensor
    """
    global csp
    check_engine()
    csp.variables.append(csp.int(size=None, deep=dimensions))
    return csp.variables[-1]


def linear(is_real=False):
    """
    Create a linear variable.
    :param is_real: If true, the variable is a real number if not an integer.
    :return: The new variable.
    """
    global csp
    check_engine()
    csp.mips.append(Linear(csp, len(csp.mips), is_real=is_real))
    return csp.mips[-1]


def maximize(objective, solve=True, lp_path=''):
    """
    Maximize the objective, according to the current linear constrains.
    :param objective: An standard linear expression.
    :param solve: Used to render lp file solve by PIXIE if True else only render lp file.
    :param lp_path: The path for the model.
    :return: the values of the model in order of variable creation.
    """
    global csp
    ints = []
    for var in csp.mips:
        if var.is_real:
            ints.append(0)
        else:
            ints.append(1)
    csp.set_integer_condition(ints)
    opt, result = csp.maximize(objective, solve, lp_path)
    if solve:
        for v, r in zip(csp.mips, result):
            if not v.is_real:
                v.value = int(r + 0.5)
            else:
                v.value = r
        return opt
    return False


def minimize(objective, solve=True, lp_path=''):
    """
    Minimize the objective, according to the current linear constrains.
    :param objective: An standard linear expression.
    :param solve: Used to render lp file solve by PIXIE if True else only render lp file.
    :param lp_path: The path for the model.
    :return: the values of the model in order of variable creation.
    """
    global csp
    ints = []
    for var in csp.mips:
        if var.is_real:
            ints.append(0)
        else:
            ints.append(1)
    csp.set_integer_condition(ints)
    opt, result = csp.minimize(objective, solve, lp_path)
    if solve:
        for v, r in zip(csp.mips, result):
            if not v.is_real:
                v.value = int(r + 0.5)
            else:
                v.value = r
        return opt
    return None


def clear(lst):
    """
    Clear a list of integers, used with optimization routines.
    :param lst: The coherent list of integers to clear.
    """
    for x in lst:
        x.clear()


def rotate(x, k):
    """
    Rotate an integer k places
    :param x: the integer.
    :param k: k-places.
    :return: a rotated integer.
    """
    v = integer()
    for i in range(bits()):
        assert x[[(i + k) % bits()]](0, 1) == v[[i]](0, 1)
    return v


def is_prime(p):
    """
    Indicate that p is prime.
    :param p: the integer.
    """
    global csp
    check_engine()
    assert pow(csp.one + csp.one, p, p) == csp.one + csp.one


def is_not_prime(p):
    """
    Indicate that p is not prime.
    :param p: the integer.
    """
    global csp
    check_engine()
    assert p != csp.one + csp.one
    assert pow(csp.one + csp.one, p, p) != csp.one + csp.one
