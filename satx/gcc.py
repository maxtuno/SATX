"""
The Global Constraint Catalog implementation for SAT-X.
"""

import satx


def abs_val(x, y):
    """
    Enforce the fact that the first variable is equal to the absolute value of the second variable.
    """
    assert y >= 0
    assert abs(x) == y


def all_differ_from_at_least_k_pos(k, lst):
    """
    Enforce all pairs of distinct vectors of the VECTORS collection to differ from at least K positions.
    """
    nil1 = satx.integer()
    nil2 = satx.integer()
    for V in lst:
        nil1.is_not_in(V)
        nil2.is_not_in(V)
    assert nil1 != nil2
    t = satx.tensor(dimensions=(len(lst[0]),))
    assert sum(t[[i]](0, 1) for i in range(len(lst[0]))) >= k
    for i1 in range(len(lst) - 1):
        for i2 in range(i1 + 1, len(lst)):
            for j in range(len(lst[0])):
                assert t[[j]](nil1, lst[i1][j]) != t[[j]](nil2, lst[i2][j])


def all_differ_from_at_most_k_pos(k, lst):
    """
    Enforce all pairs of distinct vectors of the VECTORS collection to differ from at most K positions.
    """
    nil1 = satx.integer()
    nil2 = satx.integer()
    for V in lst:
        nil1.is_not_in(V)
        nil2.is_not_in(V)
    assert nil1 == nil2
    t = satx.tensor(dimensions=(len(lst[0]),))
    assert sum(t[[i]](0, 1) for i in range(len(lst[0]))) >= len(lst[0]) - k
    for i1 in range(len(lst) - 1):
        for i2 in range(i1 + 1, len(lst)):
            for j in range(len(lst[0])):
                assert t[[j]](nil1, lst[i1][j]) == t[[j]](nil2, lst[i2][j])


def all_differ_from_exactly_k_pos(k, lst):
    """
    Enforce all pairs of distinct vectors of the VECTORS collection to differ from exactly K positions. Enforce K = 0 when |VECTORS| < 2.
    """
    all_differ_from_at_least_k_pos(k, lst)
    all_differ_from_at_most_k_pos(k, lst)


def all_equal(lst):
    """
    Enforce all variables of the collection
    """
    satx.apply_dual(lst, lambda x, y: x == y)


def all_different(lst):
    """
    Enforce all variables of the collection ππ°ππΈπ°π±π»π΄π to take distinct values.
    """
    satx.all_different(lst)


def element(idx, lst, val):
    """
    ππ°π»ππ΄ is equal to the πΈπ½π³π΄π-th item of ππ°π±π»π΄, i.e. ππ°π»ππ΄ = ππ°π±π»π΄[πΈπ½π³π΄π].
    """
    assert val == satx.index(idx, lst)


def gcd(x, y, z):
    """
    Enforce the fact that π is the greatest common divisor of π and π. (assume X <= Y)
    """
    if not isinstance(y, satx.Unit):
        y = satx.constant(y)
    assert 0 < x <= y
    assert z > 0
    assert z == y % x
    assert (x / z) % (y % z) == 0


def sort(lst1, lst2):
    """
    First, the variables of the collection ππ°ππΈπ°π±π»π΄π 2 correspond to a permutation of the variables of ππ°ππΈπ°π±π»π΄π 1. Second, the variables of ππ°ππΈπ°π±π»π΄π 2 are sorted in increasing order.
    """
    _, ys = satx.permutations(lst1, len(lst1))
    satx.apply_single(lst2, lambda i, t: t == ys[i], indexed=True)
    satx.apply_dual(lst2, lambda a, b: a <= b)


def sort_permutation(lst_from, lst_per, lst_to):
    """
    The variables of collection π΅ππΎπΌ correspond to the variables of collection ππΎ according to the permutation πΏπ΄ππΌπππ°ππΈπΎπ½ (i.e., π΅ππΎπΌ[i].πππ=ππΎ[πΏπ΄ππΌπππ°ππΈπΎπ½[i].πππ].πππ). The variables of collection ππΎ are also sorted in increasing order.
    """
    satx.apply_dual(lst_to, lambda a, b: a <= b)
    xs1, ys1 = satx.permutations(lst_from, len(lst_from))
    assert ys1 == lst_to
    assert lst_per == xs1


def count(val, lst, rel, lim):
    """
    Let N be the number of variables of the ππ°ππΈπ°π±π»π΄π collection assigned to value ππ°π»ππ΄; Enforce condition N ππ΄π»πΎπΏ π»πΈπΌπΈπ to hold.
    """
    t = satx.tensor(dimensions=(len(lst),))
    for i in range(len(lst)):
        assert t[[i]](0, lst[i] - val) == 0
    assert rel(sum(t[[i]](0, 1) for i in range(len(lst))), lim)
