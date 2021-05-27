import satx

if __name__ == '__main__':

    original_number_of_variables = 130209

    satx.engine(8, render_by_slime=True)

    x4 = satx.natural()
    x3 = satx.natural()
    x2 = satx.natural()
    x1 = satx.natural()

    unused_space = satx.integer(bits=original_number_of_variables)

    assert ord('0') <= x1 <= ord('9')
    assert ord('0') <= x2 <= ord('9')
    assert ord('0') <= x3 <= ord('9')
    assert ord('0') <= x4 <= ord('9')

    satx.satisfy(solve=False, cnf_path='data/limit.cnf')
