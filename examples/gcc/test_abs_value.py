import satx
import satx.gcc

satx.engine(10)

x = satx.integer()
y = satx.integer()

satx.gcc.abs_val(x, y)

while satx.satisfy():
    print(x, y)
