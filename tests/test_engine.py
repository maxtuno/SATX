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
import os
import subprocess
import sys
from textwrap import dedent

import pytest
import satx
from pathlib import Path


def test_engine_auto_cnf_path(tmp_path):
    script = tmp_path / "auto_engine.py"
    script.write_text(
        dedent(
            """
            import satx

            satx.engine(bits=4)
            print(satx.current_cnf_path())
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True, env=env
    )
    path = proc.stdout.strip()
    assert path.endswith(".cnf")
    assert os.path.basename(path) == "auto_engine.cnf"
    os.remove(path)


def test_engine_empty_string_preserved():
    satx.engine(bits=4, cnf_path="")
    assert satx.current_cnf_path() == ""
    satx.reset()


def test_version_returns_string(capsys):
    assert satx.version() == "0.4.0"
    capsys.readouterr()
