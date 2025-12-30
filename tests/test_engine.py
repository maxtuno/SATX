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
    with pytest.raises(Exception, match="No cnf file specified"):
        satx.engine(bits=4, cnf_path="")


def test_version_returns_string(capsys):
    assert satx.version() == "0.3.9"
    capsys.readouterr()
