# SATX `satx.gcc` coverage (Global Constraint Catalog)

This note tracks which “core” Global Constraint Catalog constraints are exposed as thin SATX-friendly wrappers in `satx/gcc.py`.

Legend: `[x]` implemented, `[ ]` not yet implemented / skipped.

Note: `satx.gcc` also includes helper constraints outside the core GCC list (e.g., `all_equal_except_0`, `all_differ_from_*`); see `satx/gcc.py` for the full set.

## Counting / Aggregation

- [x] `COUNT` (`count`) — lower-bound style encoding (as implemented)
- [x] `AMONG` (`among`, `among_var`)
- [x] `ATLEAST` (`at_least`)
- [x] `ATMOST` (`at_most`)
- [x] `EXACTLY` (`exactly`)
- [x] `NVALUE` (`nvalue`)
- [x] `SUM` (`sum`)
- [x] `SCALAR_PRODUCT` (`scalar_product`)

## Ordering / (In)Equality

- [x] `ALLDIFFERENT` (`all_different`)
- [x] `ALLEQUAL` (`all_equal`)
- [x] `ALLEQUAL_EXCEPT_0` (`all_equal_except_0`)
- [x] `SORT` (`sort`)
- [x] `SORT_PERMUTATION` (`sort_permutation`)
- [x] `LEX_LESS` (`lex_less`)
- [x] `LEX_LESSEQ` (`lex_lesseq`)
- [x] `LEX_GREATER` (`lex_greater`)
- [x] `LEX_GREATEREQ` (`lex_greatereq`)
- [x] `LEX_CHAIN_*` (`lex_chain_less`, `lex_chain_lesseq`, `lex_chain_greater`, `lex_chain_greatereq`)

## Membership / Indexing

- [x] `ELEMENT` (`element`)
- [x] `IN` (`in_`)
- [x] `NOTIN` (`not_in`)

## Permutation / Graph

- [x] `INVERSE` (`inverse`)
- [ ] `INVERSE_OFFSET` — not implemented (no offset convention exposed yet)
- [x] `CIRCUIT` (`circuit`)

## Arithmetic (catalog-style)

- [x] `ABS` (`abs_val`)
- [x] `GCD` (`gcd`)
- [x] `MINIMUM` (`minimum`)
- [x] `MAXIMUM` (`maximum`)

## Packing / Scheduling / Geometry

- [x] `BIN_PACKING` (`bin_packing`)
- [x] `BIN_PACKING_CAPA` (`bin_packing_capa`)
- [x] `CUMULATIVE` (`cumulative`) — discrete time-indexed decomposition (requires a finite horizon)
- [ ] `CUMULATIVES` — not implemented (multi-resource variant not exposed yet)
- [x] `DIFFN` (`diffn`)
- [ ] `DISJOINT` — not implemented (catalog name overloaded; use `diffn` for rectangles)
