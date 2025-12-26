# SATX

SATX is an exact constraint-based modeling system built on top of SAT solvers. It lets you express arithmetic,
logical, and algebraic constraints in Python while preserving discrete exactness.

## Installation

Use Python 3.10+.

```bash
# install the latest development sources directly from the SATX repository
pip install git+https://github.com/maxtuno/SATX.git

# install the package into a virtual environment for development
pip install -e .[dev]
```

You can also build source and wheel distributions locally:

```bash
python -m build
```

## Quickstart

```python
import satx

satx.engine(bits=8)

x = satx.integer()
y = satx.integer()

assert x > 0
assert y > 0
assert x + y == 7

print("SATX version:", satx.__version__)
```

Verify the installed version directly:

```bash
python -c "import satx; print(satx.__version__)"
```

## Development workflow

The repository uses a `src/` layout and [PEP 517](https://peps.python.org/pep-0517/) via `pyproject.toml`.
Run the following commands when you work on SATX:

- `make venv` – create `.venv`, upgrade pip, and install the package plus dev dependencies (pytest).
- `make test` – execute `python -m pytest`.
- `make build` – produce `dist/` with `python -m build`.

Once you publish, anyone can install straight from Git:

```bash
pip install git+https://<host>/<org>/SATX.git
```
