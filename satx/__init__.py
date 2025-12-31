"""
Copyright (c) 2012-2026 Oscar Riveros

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
The SATX system is a CNF compiler and SAT solver built into Python.
"""

from . import stdlib as _stdlib
from .stdlib import *

# Fixed-point decimals (scaled integers over Unit bit-vectors).
from .fixed import (
    Fixed,
    fixed,
    fixed_const,
    fixed_lcm_scale,
    fixed_rescale_to,
    fixed_add_rescaled,
    as_fixed,
    to_rational,
    fixed_div_exact,
    fixed_bounds,
    fixed_advice,
    as_int,
    fixed_mul_floor,
    fixed_mul_round,
    enumerate_models_fixed,
    block_fixed_solution,
)


def current_engine():
    return _stdlib.csp


__version__ = _stdlib.VERSION


if "csp" in globals():
    del csp


def __getattr__(name):
    if name == "csp":
        return _stdlib.csp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
