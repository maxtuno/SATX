import satx


def test_docs_fixed_default_smoke():
    satx.engine(bits=12, fixed_default=True, fixed_scale=100, cnf_path="tests/tmp_docs_fixed_default.cnf")
    x = satx.integer()
    y = satx.vector(size=2)
    assert x >= 0
    assert y[0] == 0
    assert satx.satisfy(solver="slime")


def test_docs_fixed_override_smoke():
    satx.engine(bits=12, fixed_default=True, fixed_scale=100, cnf_path="tests/tmp_docs_fixed_override.cnf")
    u = satx.integer(force_int=True)
    z = satx.vector(size=2, fixed=True, scale=1000)
    assert u == 3
    assert z[0] == satx.fixed_const(1.0, scale=1000)
    assert satx.satisfy(solver="slime")
