"""
Copyright (c) 2012–2026 Oscar Riveros

SATX is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

SATX is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Commercial licensing options are available.
See COMMERCIAL.md for details.
"""

"""
The standard high level library for the SATX system.
"""

import os
import sys

from .alu import *
from .gaussian import Gaussian
from .rational import Rational

csp = None
render = False
VERSION = "0.4.0"


def version():
    """
    Print the information about the system.
    """
    print('SATX The constraint modeling language for SAT solvers')
    print('Copyright (c) 2012-2026 Oscar Riveros. all rights reserved.')
    return VERSION


def check_engine():
    if csp is None:
        print('The SATX system is not initialized.')
        exit(0)


def _derive_default_cnf_path():
    candidate = None
    try:
        import __main__

        candidate = getattr(__main__, '__file__', None)
    except ImportError:  # pragma: no cover
        candidate = None
    if not candidate:
        argv0 = sys.argv[0] if sys.argv else None
        if argv0:
            candidate = argv0
    if not candidate:
        base = 'satx'
    else:
        base = os.path.splitext(os.path.basename(candidate))[0] or 'satx'
    return f'{base}.cnf'


def current_cnf_path():
    global csp
    return None if csp is None else csp.cnf


def _require_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def engine(
    bits=None,
    deep=None,
    info=False,
    cnf_path=None,
    signed=False,
    simplify=True,
    fixed_default=False,
    fixed_scale=1,
    max_fixed_pow=8,
):
    """
    Initialize or reset the SATX system.
    :param bits: Implies an $[-2^{bits}, 2^{bits})$ search space.
    :param deep: For exponentials range.
    :param info: Print the information about the system.
    :param cnf_path: Path to render the generated CNF. Defaults to '<script>.cnf'; pass '' to disable emitting.
    :param signed: Indicates use of signed integer engine
    :param fixed_default: When True, satx.integer()/vector()/matrix() default to Fixed with fixed_scale.
    :param fixed_scale: Default scale used when fixed_default is True.
    :param max_fixed_pow: Maximum allowed exponent for Fixed ** k.
    """
    global csp
    if cnf_path is None:
        cnf_path = _derive_default_cnf_path()
    if bits is None:
        bits = 32
    bits = _require_positive_int("bits", bits)
    if deep is None:
        deep = bits // 2
    deep = _require_positive_int("deep", deep)
    fixed_scale = _require_positive_int("fixed_scale", fixed_scale)
    max_fixed_pow = _require_positive_int("max_fixed_pow", max_fixed_pow)
    reset()
    csp = ALU(bits, deep, cnf_path)
    csp.signed = signed
    csp.simplify = simplify
    csp.default_is_fixed = bool(fixed_default)
    csp.default_scale = fixed_scale
    csp.max_fixed_pow = max_fixed_pow
    if info:
        version()


def integer(bits=None, scale=None, force_int=False):
    """
    Correspond to an integer.
    :param bits: The bits for the integer.
    :param scale: Optional fixed-point scale; when set, returns a satx.Fixed.
    :param force_int: When True, always return a Unit (ignore fixed_default).
    :return: An instance of Integer or Fixed.
    """
    global csp
    check_engine()
    if not isinstance(force_int, bool):
        raise TypeError(f"force_int must be a bool, got {type(force_int).__name__}")
    if force_int and scale is not None:
        raise ValueError("force_int cannot be used with scale")
    if scale is None and not force_int and getattr(csp, "default_is_fixed", False):
        scale = csp.default_scale
    if scale is None:
        csp.variables.append(csp.int(size=bits))
        return csp.variables[-1]
    _require_positive_int("scale", scale)
    raw = csp.int(size=bits)
    csp.variables.append(raw)
    from .fixed import Fixed
    return Fixed(raw, scale)


def _resolve_fixed_params(fixed, scale):
    global csp
    if fixed is not None and not isinstance(fixed, bool):
        raise TypeError(f"fixed must be a bool or None, got {type(fixed).__name__}")
    if scale is not None:
        if fixed is False:
            raise ValueError("scale cannot be used with fixed=False")
        _require_positive_int("scale", scale)
        return True, scale
    if fixed is True:
        return True, csp.default_scale
    if fixed is False:
        return False, None
    if getattr(csp, "default_is_fixed", False):
        return True, csp.default_scale
    return False, None


def constant(value, bits=None, scale=None):
    """
    Correspond to an constant.
    :param bits: The bits for the constant.
    :param value: The value of the constant.
    :param scale: Optional fixed-point scale; when set, returns a satx.Fixed.
    :return: An instance of Constant.
    """
    global csp
    check_engine()
    if scale is None:
        csp.variables.append(csp.int(size=bits, value=value))
        return csp.variables[-1]
    from .fixed import fixed_const
    return fixed_const(value, scale=scale)


def unit_le_fixed(u, x):
    from .fixed import Fixed
    if not isinstance(u, Unit) or not isinstance(x, Fixed):
        raise TypeError("unit_le_fixed expects (Unit, Fixed)")
    if u.alu is not x.raw.alu:
        raise ValueError("cannot compare values from different SATX engines")
    if u.value is not None and x.raw.value is not None:
        return u.value <= x.value
    scaled = u * x.scale
    lhs_block = scaled.block if isinstance(scaled, Unit) else u.alu.create_constant(scaled)
    bvc = u.alu.bv_sle_gate if u.alu.signed else u.alu.bv_ule_gate
    bvc(lhs_block, x.raw.block, u.alu.false)
    return u


def unit_ge_fixed(u, x):
    from .fixed import Fixed
    if not isinstance(u, Unit) or not isinstance(x, Fixed):
        raise TypeError("unit_ge_fixed expects (Unit, Fixed)")
    if u.alu is not x.raw.alu:
        raise ValueError("cannot compare values from different SATX engines")
    if u.value is not None and x.raw.value is not None:
        return u.value >= x.value
    scaled = u * x.scale
    lhs_block = scaled.block if isinstance(scaled, Unit) else u.alu.create_constant(scaled)
    bvc = u.alu.bv_sle_gate if u.alu.signed else u.alu.bv_ule_gate
    bvc(x.raw.block, lhs_block, u.alu.false)
    return u


def unit_eq_fixed(u, x):
    from .fixed import Fixed
    if not isinstance(u, Unit) or not isinstance(x, Fixed):
        raise TypeError("unit_eq_fixed expects (Unit, Fixed)")
    if u.alu is not x.raw.alu:
        raise ValueError("cannot compare values from different SATX engines")
    if u.value is not None and x.raw.value is not None:
        return u.value == x.value
    scaled = u * x.scale
    lhs_block = scaled.block if isinstance(scaled, Unit) else u.alu.create_constant(scaled)
    u.alu.bv_eq_gate(lhs_block, x.raw.block, u.alu.false)
    return u


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
        return bits, subset_, complement_
    else:
        return bits, subset_


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


def vector(bits=None, size=None, is_gaussian=False, is_rational=False, *, fixed=None, scale=None):
    """
    A vector of integers.
    :param bits: The bit bits for each integer.
    :param size: The bits of the vector.
    :param is_gaussian: Indicate of is a Gaussian Integers vector.
    :param is_rational: Indicate of is a Rational vector.
    :param fixed: When True/False, override fixed_default for this vector.
    :param scale: Fixed-point scale (implies fixed=True).
    :return: An instance of vector.
    """
    global csp
    check_engine()
    if is_rational:
        return [rational() for _ in range(size)]
    if is_gaussian:
        return [gaussian() for _ in range(size)]
    use_fixed, fixed_scale = _resolve_fixed_params(fixed, scale)
    array_ = csp.array(size=bits, dimension=size)
    csp.variables += array_
    if use_fixed:
        from .fixed import Fixed
        return [Fixed(item, fixed_scale) for item in array_]
    return array_


def matrix(bits=None, dimensions=None, is_gaussian=False, is_rational=False, *, fixed=None, scale=None):
    """
    A matrix of integers.
    :param bits: The bit bits for each integer.
    :param dimensions: An tuple with the dimensions for the Matrix (n, m).
    :param is_gaussian: Indicate of is a Gaussian Integers vector.
    :param is_rational: Indicate of is a Rational Matrix.
    :return: An instance of Matrix.
    """
    global csp
    check_engine()
    matrix_ = []
    use_fixed, fixed_scale = _resolve_fixed_params(fixed, scale)
    for i in range(dimensions[0]):
        row = []
        for j in range(dimensions[1]):
            if is_rational:
                x = integer(bits=bits, force_int=True)
                y = integer(bits=bits, force_int=True)
                row.append(Rational(x, y))
            elif is_gaussian:
                x = integer(bits=bits, force_int=True)
                y = integer(bits=bits, force_int=True)
                row.append(Gaussian(x, y))
            else:
                if use_fixed:
                    raw = csp.int(size=bits)
                    csp.variables.append(raw)
                    from .fixed import Fixed
                    row.append(Fixed(raw, fixed_scale))
                else:
                    csp.variables.append(integer(bits=bits, force_int=True))
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
    xs = vector(size=n, fixed=False)
    ys = vector(size=n, fixed=False)
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
    xs = vector(size=n, fixed=False)
    ys = vector(size=n, fixed=False)
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
    xs = vector(size=n, fixed=False)
    ys = vector(size=n, fixed=False)
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
    for item in lst:
        raw = getattr(item, "raw", None)
        scale = getattr(item, "scale", None)
        if isinstance(raw, Unit) and isinstance(scale, int):
            raw.is_in([0, scale])
        else:
            target = raw if isinstance(raw, Unit) else item
            assert 0 <= target <= 1


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
    if csp.signed:
        return (1 << (csp.bits - 1)) - 1
    return (1 << csp.bits) - 1


def element(item, data):
    """
    Ensure that the element i is on the data, on the position index.
    :param item: The element
    :param data: The data
    :return: The position of element
    """
    global csp
    check_engine()
    ith = integer(force_int=True)
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
    item = integer(force_int=True)
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
        return Gaussian(integer(force_int=True), integer(force_int=True))
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
        return Rational(integer(force_int=True), integer(force_int=True))
    return Rational(x, y)


def exact_div(x, y):
    """
    Exact integer division: returns q such that x == q*y and x % y == 0.
    """
    check_engine()
    if not isinstance(x, Unit):
        x = constant(x)
    if not isinstance(y, Unit):
        y = constant(y)
    assert y != 0
    output_block = csp.create_block()
    csp.bv_lud_gate(x.block, y.block, output_block, csp.zero.block)
    q = Unit(csp, block=output_block)
    csp.variables.append(q)
    return q


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
# //                        oscar.riveros@satx.science                        //
# //                                                                           //
# //   without any restriction, Oscar Riveros reserved rights, patents and     //
# //  commercialization of this knowledge or derived directly from this work.  //
# ///////////////////////////////////////////////////////////////////////////////
def hess_sequence(n, oracle, fast=False, cycles=1, target=0, seq=None):
    """
    HESS Algorithm is a Universal Black Box Optimizer (sequence version).
    :param n: The size of sequence.
    :param oracle: The oracle, this output a number and input a sequence.
    :param fast: More fast less accuracy.
    :param cycles: How many times the HESS algorithm is executed.
    :param target: Any value less than this terminates the execution.
    :param seq: External sequence if not set default sequence is used (1..n)
    :return optimized sequence.
    """
    import hashlib
    import math

    db = []

    if seq is not None:
        xs = seq
    else:
        xs = list(range(n))
    glb = math.inf
    opt = xs[:]

    def __inv(i, j, xs):
        while i < j:
            xs[i], xs[j] = xs[j], xs[i]
            i += 1
            j -= 1

    def __next_orbit(xs):
        for i in range(len(xs)):
            for j in range(len(xs)):
                key = hashlib.sha256(bytes(xs)).hexdigest()
                if key not in db:
                    db.append(key)
                    db.sort()
                    return True
                __inv(min(i, j), max(i, j), xs)
        return False

    top = glb
    for _ in range(cycles):
        if fast:
            while __next_orbit(xs):
                glb = math.inf
                anchor = top
                for i in range(len(xs) - 1):
                    for j in range(i + 1, len(xs)):
                        __inv(min(i, j), max(i, j), xs)
                        loc = oracle(xs)
                        if loc < glb:
                            glb = loc
                            if glb < top:
                                top = glb
                                opt = xs[:]
                                if top <= target:
                                    return opt
                        elif loc > glb:
                            __inv(min(i, j), max(i, j), xs)
                if top == anchor:
                    break
        else:
            while __next_orbit(xs):
                glb = math.inf
                anchor = top
                for i in range(len(xs)):
                    for j in range(len(xs)):
                        __inv(min(i, j), max(i, j), xs)
                        loc = oracle(xs)
                        if loc < glb:
                            glb = loc
                            if glb < top:
                                top = glb
                                opt = xs[:]
                                if top <= target:
                                    return opt
                        elif loc > glb:
                            __inv(min(i, j), max(i, j), xs)
                if top == anchor:
                    break
    return opt


# ///////////////////////////////////////////////////////////////////////////////
# //        Copyright (c) 2012-2020 Oscar Riveros. all rights reserved.        //
# //                        oscar.riveros@satx.science                        //
# //                                                                           //
# //   without any restriction, Oscar Riveros reserved rights, patents and     //
# //  commercialization of this knowledge or derived directly from this work.  //
# ///////////////////////////////////////////////////////////////////////////////
def hess_binary(n, oracle, fast=False, cycles=1, target=0, seq=None):
    """
    HESS Algorithm is a Universal Black Box Optimizer (binary version).
    :param n: The size of bit vector.
    :param oracle: The oracle, this output a number and input a bit vector.
    :param fast: More fast some times less accuracy.
    :param cycles: How many times the HESS algorithm is executed.
    :param target: Any value less than this terminates the execution.
    :param seq: External sequence if not set default sequence is used (1..n)
    :return optimized sequence.
    """
    import hashlib
    import math

    db = []

    if seq is not None:
        xs = seq
    else:
        xs = [False] * n
    glb = math.inf
    opt = xs[:]

    def __next_orbit(xs):
        for i in range(len(xs)):
            key = hashlib.sha256(bytes(xs)).hexdigest()
            if key not in db:
                db.append(key)
                db.sort()
                return True
            xs[i] = not xs[i]
        return False

    top = glb
    for _ in range(cycles):
        if fast:
            while __next_orbit(xs):
                glb = math.inf
                anchor = top
                for i in range(len(xs)):
                    xs[i] = not xs[i]
                    loc = oracle(xs)
                    if loc < glb:
                        glb = loc
                        if glb < top:
                            top = glb
                            opt = xs[:]
                            if top <= target:
                                return opt
                    elif loc > glb:
                        xs[i] = not xs[i]
                if top == anchor:
                    break
        else:
            while __next_orbit(xs):
                glb = math.inf
                anchor = top
                for i in range(len(xs)):
                    xs[i] = not xs[i]
                    loc = oracle(xs)
                    if loc < glb:
                        glb = loc
                        if glb < top:
                            top = glb
                            opt = xs[:]
                            if top <= target:
                                return opt
                    elif loc > glb:
                        xs[i] = not xs[i]
                if top == anchor:
                    break
    return opt


# ///////////////////////////////////////////////////////////////////////////////
# //        Copyright (c) 2012-2020 Oscar Riveros. all rights reserved.        //
# //                        oscar.riveros@satx.science                        //
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
    import hashlib
    import math

    db = []

    glb = math.inf
    opt = xs[:]

    def __next_orbit(xs):
        for i in range(len(xs)):
            for j in range(len(xs)):
                key = hashlib.sha256(bytes(xs)).hexdigest()
                if key not in db:
                    db.append(key)
                    db.sort()
                    return True
                g(min(i, j), max(i, j), xs)
                f(min(i, j), max(i, j), xs)
        return False

    top = glb
    for _ in range(cycles):
        if fast:
            glb = math.inf
            anchor = top
            while __next_orbit(xs):
                for i in range(len(xs) - 1):
                    for j in range(i + 1, len(xs)):
                        f(min(i, j), max(i, j), xs)
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
                            g(min(i, j), max(i, j), xs)
                if top == anchor:
                    break
        else:
            while __next_orbit(xs):
                glb = math.inf
                anchor = top
                for i in range(len(xs)):
                    for j in range(len(xs)):
                        f(min(i, j), max(i, j), xs)
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
                            g(min(i, j), max(i, j), xs)
                if top == anchor:
                    break
    return opt


# ///////////////////////////////////////////////////////////////////////////////
# //        Copyright (c) 2012-2020 Oscar Riveros. all rights reserved.        //
# //                        oscar.riveros@satx.science                        //
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


def clear(lst):
    """
    Clear a list of integers, used with optimization routines.
    :param lst: The coherent list of integers to clear.
    """
    for x in lst:
        target = getattr(x, "raw", x)
        target.clear()


def rotate(x, k):
    """
    Rotate an integer k places
    :param x: the integer.
    :param k: k-places.
    :return: a rotated integer.
    """
    v = integer(force_int=True)
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


def satisfy(solver, params='', log=False):
    """
    Solve with external solver.
    :param solver: The external solver.
    :param params: Parameters passed to external solver.
    :return: True if SAT else False.
    """
    global csp, render
    import subprocess
    import tempfile
    import uuid
    check_engine()
    if csp.cnf == '':
        base = os.path.join(tempfile.gettempdir(), f"satx_{uuid.uuid4().hex}")
        cnf_path = f"{base}.cnf"
        mod_path = f"{base}.mod"
        try:
            with open(cnf_path, 'w', encoding="utf8", errors='ignore') as file:
                for clause in getattr(csp, "_cnf_clauses", []):
                    file.write(' '.join(list(map(str, clause))) + ' 0\n')

            header = 'p cnf {} {}'.format(csp.number_of_variables, csp.number_of_clauses).rstrip('\r\n')
            with open(cnf_path, 'r+', encoding="utf8", errors='ignore') as file:
                content = file.read()
                file.seek(0, 0)
                file.write(header + '\n' + content)
                file.truncate()

            with open(mod_path, 'w', encoding="utf8", errors='ignore') as file:
                proc = subprocess.Popen('{0} {1}.cnf {2}'.format(solver, base, params), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for stdout_line in iter(proc.stdout.readline, ''):
                    if not stdout_line:
                        break
                    try:
                        line = stdout_line.decode()
                        file.write(line)
                        if log:
                            print(line, end='')
                    except:
                        pass
                proc.stdout.close()

            with open(mod_path, 'r') as mod:
                lines = ''
                for line in mod.readlines():
                    if line.startswith('v '):
                        lines += line.strip('v ').strip('\n') + ' '
                if len(lines) > 0:
                    model = list(map(int, lines.strip(' ').split(' ')))
                    for arg in csp.variables:
                        if isinstance(arg, Unit):
                            ds = ''.join(map(str, [int(model[abs(bit) - 1] * bit > 0) for bit in arg.block[::-1]]))
                            if csp.signed:
                                if ds[0] == '1':
                                    arg.value = -int(''.join(['0' if d == '1' else '1' for d in ds[1:]]), 2) - 1
                                else:
                                    arg.value = int(ds[1:], 2)
                            else:
                                arg.value = int(ds, 2)
                            del arg.bin[:]
                    block_clause = [-int(literal) for literal in model]
                    if getattr(csp, "_cnf_clauses", None) is not None:
                        csp._cnf_clauses.append(block_clause)
                    csp.number_of_clauses += 1
                    return True
            return False
        finally:
            for path in (cnf_path, mod_path):
                try:
                    os.remove(path)
                except Exception:
                    pass
    if '.' not in csp.cnf:
        raise Exception('CNF has no extension.')
    try:
        key = csp.cnf[:csp.cnf.index('.')]
    except Exception as ex:
        print('No .cnf extension found. {}'.format(ex))
        key = csp.cnf

    if getattr(csp, 'cnf_file', None) is not None and not csp.cnf_file.closed:
        csp.cnf_file.close()

    header = 'p cnf {} {}'.format(csp.number_of_variables, csp.number_of_clauses).rstrip('\r\n')
    if not render:
        render = True
        with open(csp.cnf, 'r+', encoding="utf8", errors='ignore') as file:
            content = file.read()
            file.seek(0, 0)
            file.write(header + '\n' + content)
            file.truncate()
    else:
        with open(csp.cnf, 'r+', encoding="utf8", errors='ignore') as file:
            content = file.read()
            _, _, rest = content.partition('\n')
            file.seek(0, 0)
            file.write(header + '\n' + rest)
            file.truncate()

    csp.cnf_file = open(csp.cnf, 'a', encoding="utf8", errors='ignore')
    with open('{}.mod'.format(key), 'w', encoding="utf8", errors='ignore') as file:
        proc = subprocess.Popen('{0} {1}.cnf {2}'.format(solver, key, params), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for stdout_line in iter(proc.stdout.readline, ''):
            if not stdout_line:
                break
            try:
                line = stdout_line.decode()
                file.write(line)
                if log:
                    print(line, end='')
            except:
                pass
        proc.stdout.close()
    with open('{}.mod'.format(key), 'r') as mod:
        lines = ''
        for line in mod.readlines():
            if line.startswith('v '):
                lines += line.strip('v ').strip('\n') + ' '
        if len(lines) > 0:
            model = list(map(int, lines.strip(' ').split(' ')))
            for arg in csp.variables:
                if isinstance(arg, Unit):
                    ds = ''.join(map(str, [int(model[abs(bit) - 1] * bit > 0) for bit in arg.block[::-1]]))
                    # Signed integers are two's complement over `csp.bits` bits:
                    # msb is sign (1 => negative), range [-2^(n-1), 2^(n-1)-1].
                    # For negatives: value = lower_bits - 2^(n-1) (sign-extended).
                    if csp.signed:
                        if ds[0] == '1':
                            arg.value = -int(''.join(['0' if d == '1' else '1' for d in ds[1:]]), 2) - 1
                        else:
                            arg.value = int(ds[1:], 2)
                    else:
                        arg.value = int(ds, 2)
                    del arg.bin[:]
            csp.cnf_file.write(' '.join([str(-int(literal)) for literal in model]) + '\n')
            csp.cnf_file.flush()
            csp.number_of_clauses += 1

            csp.cnf_file.close()
            header = 'p cnf {} {}'.format(csp.number_of_variables, csp.number_of_clauses).rstrip('\r\n')
            with open(csp.cnf, 'r+', encoding="utf8", errors='ignore') as file:
                content = file.read()
                _, _, rest = content.partition('\n')
                file.seek(0, 0)
                file.write(header + '\n' + rest)
                file.truncate()
            csp.cnf_file = open(csp.cnf, 'a', encoding="utf8", errors='ignore')
            return True
    return False


def reset():
    global csp, render
    """
    Use this with external on optimization routines.
    :param key: key of the external problem
    :return:
    """
    import os
    if csp is not None:
        if getattr(csp, 'cnf_file', None) is not None and not csp.cnf_file.closed:
            csp.cnf_file.close()
        if os.path.exists(csp.cnf):
            os.remove(csp.cnf)
    render = False
