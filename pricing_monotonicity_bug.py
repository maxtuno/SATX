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

# CNF emitida a archivo; usa tu solver "slime" (o el que tengas con salida tipo 'v ...').
satx.engine(bits=10, cnf_path="pricing_monotonicity_bug.cnf", signed=False, simplify=True)

FREE = 100
THRESH = 200
MAXU = 300

base  = satx.fixed_const(10,   scale=100)   # $10.00
rate1 = satx.fixed_const(0.10, scale=100)   # $0.10 por llamada
rate2 = satx.fixed_const(0.01, scale=100)   # $0.01 por llamada

def bill_bug(u: satx.Unit) -> satx.Fixed:
    alu = u.alu

    # u > FREE  <=>  not(u <= FREE)
    le_free = alu.bv_ule_gate(u.block, alu.create_constant(FREE))   # lit: (u <= FREE)
    gt_free = -le_free                                              # lit: (u > FREE)

    over = (u - FREE).iff(gt_free, 0)                               # max(0, u-FREE)
    charge_low = over * rate1                                       # Fixed

    # lit: (u <= THRESH)
    le_thresh = alu.bv_ule_gate(u.block, alu.create_constant(THRESH))

    # BUG: si u>THRESH, cobra SOLO (u-THRESH)*rate2 y OLVIDA el tramo anterior
    charge_high = (u - THRESH) * rate2                              # Fixed

    charge_raw = charge_low.raw.iff(le_thresh, charge_high.raw)     # if u<=THRESH -> low else high
    return base + satx.Fixed(charge_raw, scale=100)

u1 = satx.integer(bits=10, force_int=True)
u2 = satx.integer(bits=10, force_int=True)

assert u1 <= MAXU
assert u2 <= MAXU
assert u1 < u2

c1 = bill_bug(u1)
c2 = bill_bug(u2)

# BUSCAMOS violación de monotonicidad: u1<u2 pero c1>c2
assert c1 > c2

if satx.satisfy(solver="slime", params="", log=False):
    print("SAT: contraejemplo encontrado")
    print("u1 =", u1.value, "cost1 =", c1)
    print("u2 =", u2.value, "cost2 =", c2)
else:
    print("UNSAT: no existe contraejemplo (monotonicidad certificada en el rango)")
