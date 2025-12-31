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
import satx
import satx.stdlib as stdlib


def test_csp_tracks_current_engine():
    satx.reset()
    stdlib.csp = None
    assert satx.csp is None
    satx.engine(bits=8, cnf_path="")
    assert satx.csp is not None
    ref = satx.csp
    satx.engine(bits=16, cnf_path="")
    assert satx.csp is not None
    assert satx.csp is not ref
    assert satx.current_engine() is satx.csp
    satx.reset()
