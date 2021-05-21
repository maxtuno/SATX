import satx

# https://www.sat-x.io/2021/04/16/cambridge-interview-question/
# ref: https://www.youtube.com/watch?v=uHUFnSBL4rw (external youtube video)

opt = 0
while True:
    satx.engine(11, cnf='cambridge_interview_question.cnf')

    a = satx.integer()
    b = satx.integer()
    c = satx.integer()
    d = satx.integer()

    satx.apply_single([a, b, c, d], lambda x: x > 0)

    assert a + b + c + d == 63
    assert a * b + b * c + c * d > opt

    if satx.external_satisfy(solver='java -jar -Xmx4g blue.jar'):
        opt = a * b + b * c + c * d
        print(opt, a, b, c, d)
        satx.external_reset()  # !!!
    else:
        break
