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

satx.engine(bits=8, cnf_path="legal_loophole.cnf", signed=False, simplify=True)  # :contentReference[oaicite:3]{index=3}
alu = satx.csp  # stdlib.csp :contentReference[oaicite:4]{index=4}

def bit():
    return satx.integer(bits=1, force_int=True)  # 1-bit Unit :contentReference[oaicite:5]{index=5}

def lit(u):  # 1-bit Unit -> SAT literal
    return u.block[0]

def require_true(L):   alu.add_block([L])    # :contentReference[oaicite:6]{index=6}
def require_false(L):  alu.add_block([-L])

def motion_pass(pI, pA, pB, vI, vA, vB):
    # presence patterns (quorum>=2 among 3 members)
    I, A, B = lit(pI), lit(pA), lit(pB)
    yI, yA, yB = lit(vI), lit(vA), lit(vB)

    pat_IA  = alu.and_gate([ I,  A, -B])  # Investor + FounderA only
    pat_IB  = alu.and_gate([ I, -A,  B])  # Investor + FounderB only
    pat_AB  = alu.and_gate([-I,  A,  B])  # both founders only
    pat_IAB = alu.and_gate([ I,  A,  B])  # all 3

    quorum = alu.or_gate([pat_IA, pat_IB, pat_AB, pat_IAB])
    require_true(quorum)

    # C1+C2: with 2 present and Chair=Investor, pass iff Investor votes YES (tie-break)
    pass_IA = alu.and_gate([pat_IA, yI])
    pass_IB = alu.and_gate([pat_IB, yI])

    # founders-only meeting: with 2 present, majority => both YES
    pass_AB = alu.and_gate([pat_AB, alu.and_gate([yA, yB])])

    # all 3 present: majority => at least 2 YES (no ties)
    two_of_three = alu.or_gate([
        alu.and_gate([yI, yA]),
        alu.and_gate([yI, yB]),
        alu.and_gate([yA, yB]),
    ])
    pass_IAB = alu.and_gate([pat_IAB, two_of_three])

    return alu.or_gate([pass_IA, pass_IB, pass_AB, pass_IAB])

# “realidad” mínima: runway<60 habilita emergencia
runway = satx.integer(bits=7, force_int=True)
assert runway <= 59

# 2 reuniones: declarar y renovar
pI0,pA0,pB0 = bit(),bit(),bit()
vI0,vA0,vB0 = bit(),bit(),bit()

pI1,pA1,pB1 = bit(),bit(),bit()
vI1,vA1,vB1 = bit(),bit(),bit()

# Founders SIEMPRE votan NO (quieres probar que aun así pierden)
assert vA0 == 0; assert vB0 == 0
assert vA1 == 0; assert vB1 == 0
# Investor SIEMPRE vota YES
assert vI0 == 1
assert vI1 == 1

declare_pass = motion_pass(pI0,pA0,pB0, vI0,vA0,vB0)
renew_pass   = motion_pass(pI1,pA1,pB1, vI1,vA1,vB1)

# Exigimos que ambas pasen (Emergency declarado y renovado)
require_true(declare_pass)
require_true(renew_pass)

if satx.satisfy(solver="slime", params="", log=False):  # :contentReference[oaicite:7]{index=7}
    print("SAT: loophole encontrado")
    print("Meeting0 presence I/A/B:", pI0.value, pA0.value, pB0.value, " votes:", vI0.value, vA0.value, vB0.value)
    print("Meeting1 presence I/A/B:", pI1.value, pA1.value, pB1.value, " votes:", vI1.value, vA1.value, vB1.value)
else:
    print("UNSAT: no existe loophole (en este modelo)")
