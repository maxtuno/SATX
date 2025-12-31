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

import re
from pathlib import Path

import satx


def _read_setup_version():
    content = Path(__file__).resolve().parents[1].joinpath("setup.py").read_text(
        encoding="utf-8", errors="replace"
    )
    literal = re.search(r"^\s*version\s*=\s*(['\"])([^'\"]+)\1", content, re.M)
    if literal:
        return literal.group(2)
    if not re.search(r"^\s*version\s*=\s*_read_version\(\)\s*,?\s*$", content, re.M):
        raise AssertionError("setup.py does not define version in an expected way")
    return satx.stdlib.VERSION


def test_version_consistency():
    versions = {
        satx.__version__,
        satx.version(),
        satx.stdlib.VERSION,
        _read_setup_version(),
    }
    assert len(versions) == 1
    assert versions.pop() == "0.4.0"
