#!/usr/bin/env python3
"""
problems_satx.py

A small benchmark / demo suite of classic Optimization, Constraint Programming,
and Number Theory / Diophantine problems implemented with SATX + GCC wrappers.

Usage examples:
  python problems_satx.py nqueens --n 8
  python problems_satx.py knapsack --weights 3,4,5,9 --values 4,5,6,10 --capacity 10
  python problems_satx.py tsp --n 8 --seed 1 --max-cost 30
  python problems_satx.py subset-sum --universe 3,5,7,9 --target 12
  python problems_satx.py expdio
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import satx
import satx.gcc as gcc


# ---------------------------
# Utilities
# ---------------------------

def parse_int_list(csv: str) -> List[int]:
    csv = csv.strip()
    if not csv:
        return []
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


def cnf_stats(cnf_path: str) -> Optional[Tuple[int, int]]:
    """
    Best-effort parse of DIMACS header: 'p cnf <vars> <clauses>'.
    Returns (vars, clauses) if found, else None.
    """
    p = Path(cnf_path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(200):  # header should be early
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("p cnf "):
                    parts = line.split()
                    if len(parts) >= 4:
                        return int(parts[2]), int(parts[3])
        return None
    except Exception:
        return None


def print_solution(title: str, cnf_path: str, solver: str, payload: dict) -> None:
    stats = cnf_stats(cnf_path)
    print(f"\n== {title} ==")
    print(f"solver: {solver}")
    print(f"cnf:    {cnf_path}")
    if stats:
        v, c = stats
        print(f"cnf p-line: vars={v} clauses={c}")
    for k, v in payload.items():
        print(f"{k}: {v}")


# ---------------------------
# Optimization / CP Problems
# ---------------------------

def solve_knapsack(weights: List[int], values: List[int], capacity: int, bits: int, cnf: str, solver: str) -> None:
    if len(weights) != len(values):
        raise ValueError("weights and values must have same length")

    satx.engine(bits=bits, cnf_path=cnf)
    n = len(weights)

    pick = satx.vector(size=n)
    satx.all_binaries(pick)

    wsum = sum(weights[i] * pick[i] for i in range(n))
    vsum = sum(values[i] * pick[i] for i in range(n))
    assert wsum <= capacity

    best_v = None
    best_pick = None

    # Maximize vsum via iterative tightening
    while satx.satisfy(solver=solver):
        best_v = vsum.value
        best_pick = [v.value for v in pick]

        satx.clear(pick + [wsum, vsum])
        assert vsum > best_v

    print_solution("Knapsack 0/1 (maximize value)", cnf, solver, {"best_value": best_v, "pick": best_pick})


def solve_subset_sum(universe: List[int], target: int, bits: int, cnf: str, solver: str) -> None:
    satx.engine(bits=bits, cnf_path=cnf)
    bits_var, subset = satx.subsets(universe)
    assert sum(subset) == target
    ok = satx.satisfy(solver=solver)
    if not ok:
        print_solution("Subset Sum", cnf, solver, {"sat": False})
        return
    chosen = [v.value for v in subset]
    print_solution("Subset Sum", cnf, solver, {"sat": True, "target": target, "subset": chosen, "selection_bits": bits_var.value})


def solve_partition(data: List[int], bits: int, cnf: str, solver: str) -> None:
    satx.engine(bits=bits, cnf_path=cnf)
    bits_var, sub, com = satx.subsets(data, complement=True)
    assert sum(sub) == sum(com)
    ok = satx.satisfy(solver=solver)
    if not ok:
        print_solution("Balanced Partition", cnf, solver, {"sat": False})
        return
    left = [v.value for v in sub]
    right = [v.value for v in com]
    print_solution("Balanced Partition", cnf, solver, {"sat": True, "A": left, "B": right, "selection_bits": bits_var.value})


def solve_kcolor(n: int, edges: List[Tuple[int, int]], k: int, bits: int, cnf: str, solver: str) -> None:
    satx.engine(bits=bits, cnf_path=cnf)
    c = satx.vector(size=n)
    for i in range(n):
        assert 0 <= c[i] < k
    for (u, v) in edges:
        assert c[u] != c[v]
    ok = satx.satisfy(solver=solver)
    if not ok:
        print_solution(f"{k}-Coloring", cnf, solver, {"sat": False})
        return
    colors = [x.value for x in c]
    print_solution(f"{k}-Coloring", cnf, solver, {"sat": True, "colors": colors})


def solve_nqueens(n: int, bits: int, cnf: str, solver: str) -> None:
    satx.engine(bits=bits, cnf_path=cnf)
    col = satx.vector(size=n)
    for r in range(n):
        assert 0 <= col[r] < n

    gcc.all_different(col)

    for r1 in range(n):
        for r2 in range(r1 + 1, n):
            dr = r2 - r1
            assert col[r1] + dr != col[r2]
            assert col[r1] - dr != col[r2]

    ok = satx.satisfy(solver=solver)
    if not ok:
        print_solution("N-Queens", cnf, solver, {"sat": False})
        return
    sol = [v.value for v in col]
    print_solution("N-Queens", cnf, solver, {"sat": True, "cols_by_row": sol})


def solve_tsp(n: int, max_cost: int, seed: int, bits: int, cnf: str, solver: str) -> None:
    rng = random.Random(seed)
    cost = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                cost[i][j] = 0
            else:
                cost[i][j] = rng.randint(1, max_cost)

    satx.engine(bits=bits, cnf_path=cnf)

    succ = satx.vector(size=n)
    for i in range(n):
        assert 0 <= succ[i] < n
        assert succ[i] != i

    gcc.circuit(succ)

    # tour_cost = sum(cost[i][succ[i]]) implemented as table-index per row
    tour_cost = sum(satx.index(succ[i], cost[i]) for i in range(n))

    best_cost = None
    best_succ = None

    # Minimize tour_cost via iterative tightening
    while satx.satisfy(solver=solver):
        best_cost = tour_cost.value
        best_succ = [v.value for v in succ]

        satx.clear(succ + [tour_cost])
        assert tour_cost < best_cost

    print_solution("TSP (minimize tour cost) via CIRCUIT", cnf, solver, {
        "n": n,
        "seed": seed,
        "max_cost": max_cost,
        "best_cost": best_cost,
        "succ": best_succ,
        "cost_matrix": cost,
    })


def solve_binpack(items: List[int], capacity: int, n_bins: int, bits: int, cnf: str, solver: str) -> None:
    satx.engine(bits=bits, cnf_path=cnf)

    n = len(items)
    b = satx.vector(size=n)
    for i in range(n):
        assert 0 <= b[i] < n_bins

    loads = satx.vector(size=n_bins)
    for j in range(n_bins):
        assert 0 <= loads[j] <= capacity

    gcc.bin_packing_capa(bins=b, sizes=items, loads=loads, capa=capacity)

    ok = satx.satisfy(solver=solver)
    if not ok:
        print_solution("Bin Packing (capacity)", cnf, solver, {"sat": False})
        return

    assign = [v.value for v in b]
    ld = [v.value for v in loads]
    print_solution("Bin Packing (capacity)", cnf, solver, {"sat": True, "assign": assign, "loads": ld})


def solve_rectpack(widths: List[int], heights: List[int], W: int, H: int, bits: int, cnf: str, solver: str) -> None:
    if len(widths) != len(heights):
        raise ValueError("widths and heights must have same length")
    satx.engine(bits=bits, cnf_path=cnf)
    n = len(widths)

    x = satx.vector(size=n)
    y = satx.vector(size=n)

    for i in range(n):
        assert 0 <= x[i] <= W - widths[i]
        assert 0 <= y[i] <= H - heights[i]

    gcc.diffn(x=x, y=y, dx=widths, dy=heights)

    ok = satx.satisfy(solver=solver)
    if not ok:
        print_solution("Rectangle Packing (DIFFN)", cnf, solver, {"sat": False})
        return

    xs = [v.value for v in x]
    ys = [v.value for v in y]
    print_solution("Rectangle Packing (DIFFN)", cnf, solver, {"sat": True, "x": xs, "y": ys, "W": W, "H": H})


# ---------------------------
# Number Theory / Diophantine
# ---------------------------

def solve_pyth(r: int, bits: int, cnf: str, solver: str) -> None:
    satx.engine(bits=bits, cnf_path=cnf)
    x = satx.integer()
    y = satx.integer()
    assert 0 <= x <= r
    assert 0 <= y <= r
    assert x**2 + y**2 == r*r
    ok = satx.satisfy(solver=solver)
    if not ok:
        print_solution("Pythagorean (x^2+y^2=r^2)", cnf, solver, {"sat": False})
        return
    print_solution("Pythagorean (x^2+y^2=r^2)", cnf, solver, {"sat": True, "r": r, "x": x.value, "y": y.value})


def solve_factor(n: int, bits: int, cnf: str, solver: str) -> None:
    satx.engine(bits=bits, cnf_path=cnf)
    p = satx.integer()
    q = satx.integer()
    assert p > 1
    assert q > 1
    assert p <= q
    assert p * q == n
    ok = satx.satisfy(solver=solver)
    if not ok:
        print_solution("Factorization (p*q=n)", cnf, solver, {"sat": False, "n": n})
        return
    print_solution("Factorization (p*q=n)", cnf, solver, {"sat": True, "n": n, "p": p.value, "q": q.value})


def solve_expdio(bits: int, deep: int, cnf: str, solver: str) -> None:
    satx.engine(bits=bits, deep=deep, cnf_path=cnf)
    two = satx.constant(2)
    k = satx.integer()
    x = satx.integer()

    # Example: 2^k - 7 = x^2
    assert 1 <= k <= deep
    assert 0 <= x <= 1 << (bits - 1)
    assert two**k - 7 == x**2

    ok = satx.satisfy(solver=solver)
    if not ok:
        print_solution("Exponential Diophantine (2^k-7=x^2)", cnf, solver, {"sat": False})
        return
    print_solution("Exponential Diophantine (2^k-7=x^2)", cnf, solver, {"sat": True, "k": k.value, "x": x.value})


def solve_gcd_example(bits: int, cnf: str, solver: str) -> None:
    satx.engine(bits=bits, cnf_path=cnf)
    x = satx.integer()
    z = satx.integer()

    gcc.gcd(x, 10, z)   # z = gcd(x,10)

    assert 0 <= x <= 50
    assert z == 5

    ok = satx.satisfy(solver=solver)
    if not ok:
        print_solution("GCD constraint example", cnf, solver, {"sat": False})
        return
    print_solution("GCD constraint example", cnf, solver, {"sat": True, "x": x.value, "gcd(x,10)": z.value})


# ---------------------------
# CLI
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="problems_satx.py")
    p.add_argument("--solver", default="slime", help="SAT solver name/path as used by satx.satisfy (default: slime)")
    p.add_argument("--cnf", default="", help="CNF output path (default: auto per problem)")
    p.add_argument("--bits", type=int, default=12, help="SATX bit-width (default: 12)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # knapsack
    s = sub.add_parser("knapsack", help="0/1 knapsack maximize value")
    s.add_argument("--weights", required=True, help="comma-separated weights")
    s.add_argument("--values", required=True, help="comma-separated values")
    s.add_argument("--capacity", type=int, required=True)

    # subset sum
    s = sub.add_parser("subset-sum", help="subset sum equals target")
    s.add_argument("--universe", required=True, help="comma-separated ints")
    s.add_argument("--target", type=int, required=True)

    # partition
    s = sub.add_parser("partition", help="balanced partition (sum(A)=sum(B))")
    s.add_argument("--data", required=True, help="comma-separated ints")

    # kcolor
    s = sub.add_parser("kcolor", help="k-coloring")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--k", type=int, required=True)
    s.add_argument("--edges", default="", help="edges as 'u-v,u-v,...' (0-indexed). Example: 0-1,1-2,2-0")

    # nqueens
    s = sub.add_parser("nqueens", help="N-Queens")
    s.add_argument("--n", type=int, required=True)

    # tsp
    s = sub.add_parser("tsp", help="TSP using CIRCUIT, random cost matrix")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--max-cost", type=int, default=30)

    # binpack
    s = sub.add_parser("binpack", help="bin packing with capacity")
    s.add_argument("--items", required=True, help="comma-separated item sizes")
    s.add_argument("--capacity", type=int, required=True)
    s.add_argument("--bins", type=int, required=True)

    # rectpack
    s = sub.add_parser("rectpack", help="rectangle packing with DIFFN")
    s.add_argument("--widths", required=True, help="comma-separated widths")
    s.add_argument("--heights", required=True, help="comma-separated heights")
    s.add_argument("--W", type=int, required=True)
    s.add_argument("--H", type=int, required=True)

    # pyth
    s = sub.add_parser("pyth", help="pythagorean x^2+y^2=r^2")
    s.add_argument("--r", type=int, required=True)

    # factor
    s = sub.add_parser("factor", help="factorization p*q=n")
    s.add_argument("--n", type=int, required=True)

    # expdio
    s = sub.add_parser("expdio", help="exponential diophantine 2^k-7=x^2")
    s.add_argument("--deep", type=int, default=8, help="max exponent / satx.engine(deep=...) (default: 8)")

    # gcd
    sub.add_parser("gcd", help="gcd constraint example gcd(x,10)=5")

    return p


def parse_edges(spec: str) -> List[Tuple[int, int]]:
    spec = spec.strip()
    if not spec:
        return []
    edges = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Bad edge token: {part} (expected u-v)")
        u, v = part.split("-", 1)
        edges.append((int(u), int(v)))
    return edges


def default_cnf_for(cmd: str) -> str:
    return f"{cmd}.cnf"


def main(argv: List[str]) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    solver = args.solver
    bits = args.bits
    cnf = args.cnf.strip() or default_cnf_for(args.cmd)

    try:
        if args.cmd == "knapsack":
            solve_knapsack(parse_int_list(args.weights), parse_int_list(args.values), args.capacity, bits, cnf, solver)
        elif args.cmd == "subset-sum":
            solve_subset_sum(parse_int_list(args.universe), args.target, bits, cnf, solver)
        elif args.cmd == "partition":
            data = parse_int_list(args.data)
            solve_partition(data, bits, cnf, solver)
        elif args.cmd == "kcolor":
            edges = parse_edges(args.edges)
            solve_kcolor(args.n, edges, args.k, bits, cnf, solver)
        elif args.cmd == "nqueens":
            solve_nqueens(args.n, bits, cnf, solver)
        elif args.cmd == "tsp":
            solve_tsp(args.n, args.max_cost, args.seed, bits, cnf, solver)
        elif args.cmd == "binpack":
            solve_binpack(parse_int_list(args.items), args.capacity, args.bins, bits, cnf, solver)
        elif args.cmd == "rectpack":
            solve_rectpack(parse_int_list(args.widths), parse_int_list(args.heights), args.W, args.H, bits, cnf, solver)
        elif args.cmd == "pyth":
            solve_pyth(args.r, bits, cnf, solver)
        elif args.cmd == "factor":
            solve_factor(args.n, bits, cnf, solver)
        elif args.cmd == "expdio":
            solve_expdio(bits=bits, deep=args.deep, cnf=cnf, solver=solver)
        elif args.cmd == "gcd":
            solve_gcd_example(bits=bits, cnf=cnf, solver=solver)
        else:
            ap.error(f"Unknown cmd: {args.cmd}")
            return 2
        return 0
    except AssertionError as e:
        print(f"\n[UNSAT or constraint failure] {e}")
        print(f"Hint: increase --bits, tighten bounds, or check the model constraints.")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
