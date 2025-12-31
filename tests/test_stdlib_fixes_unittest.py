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
import glob
import os
import tempfile
import unittest

import satx
import satx.stdlib as stdlib
from satx.gaussian import Gaussian
from satx.rational import Rational


class StdlibFixesTest(unittest.TestCase):
    def tearDown(self):
        satx.reset()

    def test_engine_defaults_no_args(self):
        satx.engine()
        self.assertEqual(satx.bits(), 32)
        self.assertEqual(stdlib.csp.deep, (satx.bits() // 2) + 1)

    def test_engine_rejects_invalid_bits_deep(self):
        with self.assertRaises(TypeError):
            satx.engine(bits="8")
        with self.assertRaises(ValueError):
            satx.engine(bits=0)
        with self.assertRaises(ValueError):
            satx.engine(bits=8, deep=0)

    def test_cnf_path_empty_in_memory_cleanup(self):
        tmpdir = tempfile.gettempdir()
        before = set(glob.glob(os.path.join(tmpdir, "satx_*.cnf")))
        before |= set(glob.glob(os.path.join(tmpdir, "satx_*.mod")))
        satx.engine(bits=4, cnf_path="")
        x = satx.integer(force_int=True)
        _ = x == 1
        result = satx.satisfy(solver="slime")
        self.assertIsInstance(result, bool)
        after = set(glob.glob(os.path.join(tmpdir, "satx_*.cnf")))
        after |= set(glob.glob(os.path.join(tmpdir, "satx_*.mod")))
        self.assertEqual(before, after)

    def test_matrix_rational_gaussian_cells(self):
        satx.engine(bits=4, cnf_path="tests/tmp_matrix_rational.cnf")
        mtx = satx.matrix(dimensions=(2, 2), is_rational=True)
        self.assertEqual(len(mtx), 2)
        self.assertTrue(all(isinstance(cell, Rational) for row in mtx for cell in row))

        satx.engine(bits=4, cnf_path="tests/tmp_matrix_gaussian.cnf")
        mtx = satx.matrix(dimensions=(2, 2), is_gaussian=True)
        self.assertEqual(len(mtx), 2)
        self.assertTrue(all(isinstance(cell, Gaussian) for row in mtx for cell in row))

    def test_constant_cache_by_size(self):
        satx.engine(bits=8, cnf_path="tests/tmp_const_cache.cnf")
        alu = stdlib.csp
        block8 = alu.create_constant(1, size=8)
        block16 = alu.create_constant(1, size=16)
        self.assertEqual(len(block8), 8)
        self.assertEqual(len(block16), 16)
        self.assertIsNot(block8, block16)

    def test_fixed_mul_helpers_use_cache(self):
        satx.engine(bits=8, cnf_path="tests/tmp_fixed_mul_helpers_cache.cnf")
        a = satx.fixed_const(1.5, scale=10)
        b = satx.fixed_const(2.0, scale=10)
        floor_val = satx.fixed_mul_floor(a, b)
        round_val = satx.fixed_mul_round(a, b)
        self.assertIsInstance(floor_val, satx.Fixed)
        self.assertIsInstance(round_val, satx.Fixed)
        self.assertTrue(hasattr(stdlib.csp, "_fixed_mul_cache"))
        self.assertGreater(len(stdlib.csp._fixed_mul_cache), 0)
