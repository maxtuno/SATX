def test_package_imports_and_version():
    import satx

    assert satx.__version__.startswith("0.")
    assert hasattr(satx, "__version__")
