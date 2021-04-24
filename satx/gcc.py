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
    T = satx.tensor(dimensions=(len(lst[0]),))
    assert sum(T[[i]](0, 1) for i in range(len(lst[0]))) >= k
    for i1 in range(len(lst) - 1):
        for i2 in range(i1 + 1, len(lst)):
            for j in range(len(lst[0])):
                assert T[[j]](nil1, lst[i1][j]) != T[[j]](nil2, lst[i2][j])


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
    T = satx.tensor(dimensions=(len(lst[0]),))
    assert sum(T[[i]](0, 1) for i in range(len(lst[0]))) >= len(lst[0]) - k
    for i1 in range(len(lst) - 1):
        for i2 in range(i1 + 1, len(lst)):
            for j in range(len(lst[0])):
                assert T[[j]](nil1, lst[i1][j]) == T[[j]](nil2, lst[i2][j])


def all_differ_from_exactly_k_pos(k, lst):
    """
    Enforce all pairs of distinct vectors of the VECTORS collection to differ from exactly K positions. Enforce K = 0 when |VECTORS| < 2.
    """
    all_differ_from_at_least_k_pos(k, lst)
    all_differ_from_at_most_k_pos(k, lst)
