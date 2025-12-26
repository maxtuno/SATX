"""
The Global Constraint Catalog implementation for SATX.
"""

import builtins
import satx


def _as_unit(x):
    return x if isinstance(x, satx.Unit) else satx.constant(x)


def _bvc(alu):
    return alu.bv_sle_gate if alu.signed else alu.bv_ule_gate


def _eq_lit(x, other):
    alu = x.alu
    rhs_block = other.block if isinstance(other, satx.Unit) else alu.create_constant(other)
    return alu.bv_eq_gate(x.block, rhs_block)


def _in_values_lit(x, values):
    if not values:
        return x.alu.true  # constant false in SATX CNF encoding
    eqs = [_eq_lit(x, v) for v in values]
    if len(eqs) == 1:
        return eqs[0]
    return x.alu.or_gate(eqs)


def _count_in_values(lst, values):
    if len(lst) == 0:
        return satx.constant(0)
    bits = satx.tensor(dimensions=(len(lst),))
    for i, xi in enumerate(lst):
        xi = _as_unit(xi)
        in_lit = _in_values_lit(xi, values)
        xi.alu.bv_eq_gate([bits.block[i]], [in_lit], xi.alu.false)
    return builtins.sum(bits[[i]](0, 1) for i in range(len(lst)))


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
    assert builtins.sum(t[[i]](0, 1) for i in range(len(lst[0]))) >= k
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
    assert builtins.sum(t[[i]](0, 1) for i in range(len(lst[0]))) >= len(lst[0]) - k
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


def all_equal_except_0(lst):
    """
    Enforce all non-zero variables of the collection to take the same value.
    """
    if len(lst) < 2:
        return
    lst = [_as_unit(x) for x in lst]
    alu = lst[0].alu
    zc = alu.create_constant(0)
    for i in range(len(lst) - 1):
        for j in range(i + 1, len(lst)):
            a = lst[i]
            b = lst[j]
            a_is_zero = alu.bv_eq_gate(a.block, zc)
            b_is_zero = alu.bv_eq_gate(b.block, zc)
            both_nonzero = alu.and_gate([-a_is_zero, -b_is_zero])
            assert (a - b).iff(both_nonzero, 0) == 0


def all_different(lst):
    """
    Enforce all variables of the collection 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 to take distinct values.
    """
    satx.all_different(lst)


def element(idx, lst, val):
    """
    𝚅𝙰𝙻𝚄𝙴 is equal to the 𝙸𝙽𝙳𝙴𝚇-th item of 𝚃𝙰𝙱𝙻𝙴, i.e. 𝚅𝙰𝙻𝚄𝙴 = 𝚃𝙰𝙱𝙻𝙴[𝙸𝙽𝙳𝙴𝚇].
    """
    assert val == satx.index(idx, lst)


def in_(x, domain):
    """
    Enforce x to take a value in a set of values, or within an inclusive (lb, ub) interval.
    """
    x = _as_unit(x)
    if isinstance(domain, tuple) and len(domain) == 2:
        lb, ub = domain
        assert lb <= x <= ub
        return
    values = list(domain)
    assert x == satx.one_of(values)


def not_in(x, values):
    """
    Enforce x to take a value not in the given set of values.
    """
    x = _as_unit(x)
    for v in values:
        assert x != v


def among(n, lst, values):
    """
    Enforce n to be the number of variables in lst taking a value from values.
    """
    assert _count_in_values(lst, values) == n


def among_var(n, lst, values):
    """
    Enforce n (a variable) to be the number of variables in lst taking a value from values.
    """
    among(n, lst, values)


def at_least(k, lst, val):
    """
    Enforce that value val occurs at least k times in lst.
    """
    assert _count_in_values(lst, [val]) >= k


def at_most(k, lst, val):
    """
    Enforce that value val occurs at most k times in lst.
    """
    assert _count_in_values(lst, [val]) <= k


def exactly(k, lst, val):
    """
    Enforce that value val occurs exactly k times in lst.
    """
    assert _count_in_values(lst, [val]) == k


def minimum(mn, lst):
    """
    Enforce mn to be the minimum of lst.
    """
    mn = _as_unit(mn)
    lst = [_as_unit(x) for x in lst]
    assert mn == satx.one_of(lst)
    for x in lst:
        assert mn <= x


def maximum(mx, lst):
    """
    Enforce mx to be the maximum of lst.
    """
    mx = _as_unit(mx)
    lst = [_as_unit(x) for x in lst]
    assert mx == satx.one_of(lst)
    for x in lst:
        assert mx >= x


def sum(lst, rel, rhs):
    """
    Enforce rel(sum(lst), rhs) to hold.
    """
    assert rel(builtins.sum(lst), rhs)


def scalar_product(coeffs, lst, rel, rhs):
    """
    Enforce rel(sum(coeffs[i] * lst[i]), rhs) to hold.
    """
    assert rel(satx.dot(lst, coeffs), rhs)


def nvalue(n, lst):
    """
    Enforce n to be the number of distinct values taken by lst.
    """
    if len(lst) == 0:
        assert n == 0
        return
    if len(lst) == 1:
        assert n == 1
        return
    lst = [_as_unit(x) for x in lst]
    ys = satx.vector(size=len(lst))
    sort(lst, ys)

    alu = ys[0].alu
    diffs = satx.tensor(dimensions=(len(lst) - 1,))
    for i in range(1, len(lst)):
        eq_lit = alu.bv_eq_gate(ys[i].block, ys[i - 1].block)
        alu.bv_eq_gate([diffs.block[i - 1]], [-eq_lit], alu.false)

    n_distinct = 1 + builtins.sum(diffs[[i]](0, 1) for i in range(len(lst) - 1))
    assert n_distinct == n


def lex_less(a, b):
    """
    Enforce a to be lexicographically strictly smaller than b.
    """
    if len(a) == 0 and len(b) == 0:
        z = satx.integer(bits=1)
        assert z == 0
        assert z == 1
        return
    assert len(a) == len(b)
    a = [_as_unit(x) for x in a]
    b = [_as_unit(x) for x in b]
    n = len(a)

    sel = satx.tensor(dimensions=(n,))
    assert builtins.sum(sel[[i]](0, 1) for i in range(n)) == 1

    alu = a[0].alu
    bvc = _bvc(alu)
    for k in range(n):
        sk = sel.block[k]
        for i in range(k):
            assert (a[i] - b[i]).iff(sk, 0) == 0
        b_le_a = bvc(b[k].block, a[k].block)
        alu.add_block([-sk, -b_le_a])


def lex_lesseq(a, b):
    """
    Enforce a to be lexicographically smaller than or equal to b.
    """
    if len(a) == 0 and len(b) == 0:
        return
    assert len(a) == len(b)
    a = [_as_unit(x) for x in a]
    b = [_as_unit(x) for x in b]
    n = len(a)

    sel = satx.tensor(dimensions=(n + 1,))
    assert builtins.sum(sel[[i]](0, 1) for i in range(n + 1)) == 1

    alu = a[0].alu
    bvc = _bvc(alu)
    for k in range(n):
        sk = sel.block[k]
        for i in range(k):
            assert (a[i] - b[i]).iff(sk, 0) == 0
        b_le_a = bvc(b[k].block, a[k].block)
        alu.add_block([-sk, -b_le_a])

    seq = sel.block[n]
    for i in range(n):
        assert (a[i] - b[i]).iff(seq, 0) == 0


def lex_greater(a, b):
    """
    Enforce a to be lexicographically strictly greater than b.
    """
    lex_less(b, a)


def lex_greatereq(a, b):
    """
    Enforce a to be lexicographically greater than or equal to b.
    """
    lex_lesseq(b, a)


def lex_chain_less(vectors):
    """
    Enforce strict lexicographic increasing order across a list of vectors.
    """
    for i in range(len(vectors) - 1):
        lex_less(vectors[i], vectors[i + 1])


def lex_chain_lesseq(vectors):
    """
    Enforce lexicographic nondecreasing order across a list of vectors.
    """
    for i in range(len(vectors) - 1):
        lex_lesseq(vectors[i], vectors[i + 1])


def lex_chain_greater(vectors):
    """
    Enforce strict lexicographic decreasing order across a list of vectors.
    """
    for i in range(len(vectors) - 1):
        lex_greater(vectors[i], vectors[i + 1])


def lex_chain_greatereq(vectors):
    """
    Enforce lexicographic nonincreasing order across a list of vectors.
    """
    for i in range(len(vectors) - 1):
        lex_greatereq(vectors[i], vectors[i + 1])


def inverse(fwd, inv):
    """
    Enforce inv to be the inverse mapping of fwd over 0..n-1.
    """
    assert len(fwd) == len(inv)
    n = len(fwd)
    fwd = [_as_unit(x) for x in fwd]
    inv = [_as_unit(x) for x in inv]
    for i in range(n):
        assert 0 <= fwd[i] < n
        assert 0 <= inv[i] < n
    for i in range(n):
        assert satx.index(fwd[i], inv) == i
        assert satx.index(inv[i], fwd) == i


def circuit(succ):
    """
    Enforce succ to describe a single Hamiltonian circuit over 0..n-1.
    """
    n = len(succ)
    if n == 0:
        return
    succ = [_as_unit(x) for x in succ]
    tour = satx.vector(size=n)
    assert tour[0] == 0
    satx.apply_single(tour, lambda t: 0 <= t < n)
    satx.apply_dual(tour, lambda a, b: a != b)
    for k in range(n - 1):
        assert satx.index(tour[k], succ) == tour[k + 1]
    assert satx.index(tour[n - 1], succ) == tour[0]


def bin_packing(load, bin, size):
    """
    Enforce load[b] to be the sum of sizes of items assigned to bin b.
    """
    assert len(bin) == len(size)
    load = [_as_unit(x) for x in load]
    bin = [_as_unit(x) for x in bin]
    m = len(load)
    n = len(bin)
    if n == 0:
        for b in range(m):
            assert load[b] == 0
        return

    for i in range(n):
        assert 0 <= bin[i] < m

    alu = bin[0].alu
    for b in range(m):
        bits = satx.tensor(dimensions=(n,))
        for i in range(n):
            eq_lit = alu.bv_eq_gate(bin[i].block, alu.create_constant(b))
            alu.bv_eq_gate([bits.block[i]], [eq_lit], alu.false)
        assert load[b] == builtins.sum(bits[[i]](0, size[i]) for i in range(n))


def bin_packing_capa(load, bin, size, capa):
    """
    Enforce BIN_PACKING and load[b] <= capa[b] for each bin b.
    """
    load = [_as_unit(x) for x in load]
    capa = [_as_unit(x) for x in capa]
    assert len(load) == len(capa)
    bin_packing(load, bin, size)
    for b in range(len(load)):
        assert load[b] <= capa[b]


def diffn(x, y, dx, dy):
    """
    Enforce all rectangles (x[i], y[i], dx[i], dy[i]) to be pairwise non-overlapping.
    """
    assert len(x) == len(y) == len(dx) == len(dy)
    n = len(x)
    x = [_as_unit(v) for v in x]
    y = [_as_unit(v) for v in y]
    dx = [_as_unit(v) for v in dx]
    dy = [_as_unit(v) for v in dy]
    if n < 2:
        return
    alu = x[0].alu
    bvc = _bvc(alu)
    for i in range(n - 1):
        for j in range(i + 1, n):
            sep = satx.tensor(dimensions=(4,))
            assert builtins.sum(sep[[k]](0, 1) for k in range(4)) >= 1

            leq0 = bvc((x[i] + dx[i]).block, x[j].block)
            leq1 = bvc((x[j] + dx[j]).block, x[i].block)
            leq2 = bvc((y[i] + dy[i]).block, y[j].block)
            leq3 = bvc((y[j] + dy[j]).block, y[i].block)

            alu.add_block([-sep.block[0], leq0])
            alu.add_block([-sep.block[1], leq1])
            alu.add_block([-sep.block[2], leq2])
            alu.add_block([-sep.block[3], leq3])


def cumulative(start, duration, demand, limit, horizon):
    """
    Enforce a discrete-time cumulative resource constraint over t in [0, horizon).
    """
    assert len(start) == len(duration) == len(demand)
    start = [_as_unit(s) for s in start]
    demand = [_as_unit(r) for r in demand]
    limit = _as_unit(limit)
    n = len(start)
    if n == 0 or horizon <= 0:
        return

    alu = start[0].alu
    bvc = _bvc(alu)
    for i in range(n):
        assert 0 <= start[i]
        assert start[i] + duration[i] <= horizon

    for t in range(horizon):
        total = 0
        for i in range(n):
            s_le_t = bvc(start[i].block, alu.create_constant(t))
            t1_le_end = bvc(alu.create_constant(t + 1), (start[i] + duration[i]).block)
            active = alu.and_gate([s_le_t, t1_le_end])
            total += demand[i].iff(active, 0)
        assert total <= limit


def gcd(x, y, z):
    """
    Enforce the fact that 𝚉 is the greatest common divisor of 𝚇 and 𝚈. (assume X <= Y)
    """
    if not isinstance(y, satx.Unit):
        y = satx.constant(y)
    assert 0 < x <= y
    assert z > 0
    assert z == y % x
    assert satx.exact_div(x, z) % (y % z) == 0


def sort(lst1, lst2):
    """
    First, the variables of the collection 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 2 correspond to a permutation of the variables of 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 1. Second, the variables of 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 2 are sorted in increasing order.
    """
    _, ys = satx.permutations(lst1, len(lst1))
    satx.apply_single(lst2, lambda i, t: t == ys[i], indexed=True)
    satx.apply_dual(lst2, lambda a, b: a <= b)


def sort_permutation(lst_from, lst_per, lst_to):
    """
    The variables of collection 𝙵𝚁𝙾𝙼 correspond to the variables of collection 𝚃𝙾 according to the permutation 𝙿𝙴𝚁𝙼𝚄𝚃𝙰𝚃𝙸𝙾𝙽 (i.e., 𝙵𝚁𝙾𝙼[i].𝚟𝚊𝚛=𝚃𝙾[𝙿𝙴𝚁𝙼𝚄𝚃𝙰𝚃𝙸𝙾𝙽[i].𝚟𝚊𝚛].𝚟𝚊𝚛). The variables of collection 𝚃𝙾 are also sorted in increasing order.
    """
    satx.apply_dual(lst_to, lambda a, b: a <= b)
    xs1, ys1 = satx.permutations(lst_from, len(lst_from))
    assert ys1 == lst_to
    assert lst_per == xs1


def count(val, lst, rel, lim):
    """
    Let N be the number of variables of the 𝚅𝙰𝚁𝙸𝙰𝙱𝙻𝙴𝚂 collection assigned to value 𝚅𝙰𝙻𝚄𝙴; Enforce condition N 𝚁𝙴𝙻𝙾𝙿 𝙻𝙸𝙼𝙸𝚃 to hold.
    """
    t = satx.tensor(dimensions=(len(lst),))
    for i in range(len(lst)):
        assert t[[i]](0, lst[i] - val) == 0
    assert rel(builtins.sum(t[[i]](0, 1) for i in range(len(lst))), lim)
