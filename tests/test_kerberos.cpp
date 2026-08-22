// test_kerberos — ruta simbólica: circuitos CBE resueltos por el kernel SLIME de Kerberos
// (§14.5–§14.6 de la normativa)

#include <satx/satx.hpp>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>

using satx::complex;
using satx::solver::result;

namespace {

bool close(std::complex<double> a, std::complex<double> b, double tol) {
  return std::abs(a - b) <= tol;
}

}  // namespace

int main() {
  // Solver básico: unidades consistentes e inconsistentes
  {
    satx::engine e;
    e.add_unit(2);
    e.add_unit(-3);
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert(m->get(2) == true);
    assert(m->get(3) == false);
    assert(m->get(-3) == true);
    assert(m->get(1) == true);  // la variable 1 es la constante VERDADERO
  }
  {
    satx::engine e;
    e.add_unit(2);
    e.add_unit(-2);
    auto m = satx::solver::solve(e);
    assert(!m.has_value() && m.error() == result::unsat);
  }

  // x == one  (variable libre restringida por igualdad bit a bit)
  {
    satx::engine e;
    const complex<6, 2> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<6, 2>::one(e)));
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert(close(x.value(*m), {1.0, 0.0}, 1e-9));
  }

  // x + one == (2.0, 0.5) → x == (1.0, 0.5)
  {
    satx::engine e;
    const complex<6, 2> x{e};
    const auto z = x + complex<6, 2>::one(e);
    e.add_unit(satx::eq_lit(e, z, complex<6, 2>{2.0, 0.5}));
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert(close(x.value(*m), {1.0, 0.5}, 1e-9));
  }

  // i · i == −1 en la ruta simbólica (sin variables libres)
  {
    satx::engine e;
    const auto z = complex<6, 2>::i_unit(e) * complex<6, 2>::i_unit(e);
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert(close(z.value(*m), {-1.0, 0.0}, 1e-9));
  }

  // Overflow de mul → UNSAT (x·x con x forzado a 10 desborda el rango NB de W=6)
  {
    satx::engine e;
    const complex<6, 0> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<6, 0>{10, 0}));
    (void)(x * x);  // añade las restricciones de no-desborde del producto
    auto m = satx::solver::solve(e);
    assert(!m.has_value() && m.error() == result::unsat);
  }

  // División simbólica: q = x / i con x = (−1, 2); identidad q·i == x
  {
    satx::engine e;
    const complex<6, 2> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<6, 2>{-1.0, 2.0}));
    const auto q = x / complex<6, 2>::i_unit(e);
    const auto back = q * complex<6, 2>::i_unit(e);
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert(close(back.value(*m), x.value(*m), 1e-9));
  }

  // División por cero simbólica → UNSAT (restricción de divisor no nulo, ADR)
  {
    satx::engine e;
    const complex<6, 2> x{e};
    (void)(x / complex<6, 2>::zero(e));
    auto m = satx::solver::solve(e);
    assert(!m.has_value() && m.error() == result::unsat);
  }

  // q = x / one  →  q == x bajo el modelo
  {
    satx::engine e;
    const complex<6, 2> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<6, 2>{-0.75, 1.5}));
    const auto q = x / complex<6, 2>::one(e);
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert(close(q.value(*m), x.value(*m), 1e-9));
  }

  // Cruz de capas: conversión NB→TC por circuito == oracle aritmético
  {
    satx::engine e;
    const complex<6, 2> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<6, 2>{-1.25, 0.75}));  // raws (−5, 3)
    const auto tc = x.tc_real_rail(e);  // circuito NB→TC (§7.3)
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    std::uint64_t u = 0;
    for (std::size_t i = 0; i < tc.size(); ++i)
      if (m->get(tc[i])) u |= (std::uint64_t{1} << i);
    if (u >= 32) u -= 64;
    assert(static_cast<std::int64_t>(u) == -5);
  }

  // abs_sq simbólico: |x|² con im == 0 bajo el modelo (§10.7)
  // (|x|² debe caber en la caja NB de W=6: 2.0 ≤ 5.25 ✓)
  {
    satx::engine e;
    const complex<6, 2> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<6, 2>{1.0, -1.0}));
    const auto s = satx::abs_sq(x);
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert(close(s.value(*m), {2.0, 0.0}, 1e-9));
  }

  // lt_lit / le_lit simbólicos: satisfacibles e insatisfacibles (§10.7)
  {
    satx::engine e;
    const complex<6, 2> x{e}, y{e};
    e.add_unit(satx::eq_lit(e, x, complex<6, 2>{1.0, 2.0}));
    e.add_unit(satx::eq_lit(e, y, complex<6, 2>{1.0, 3.0}));
    e.add_unit(satx::lt_lit(e, x, y));
    e.add_unit(satx::le_lit(e, y, y));
    auto m = satx::solver::solve(e);
    assert(m.has_value());
  }
  {
    satx::engine e;
    const complex<6, 2> x{e}, y{e};
    e.add_unit(satx::eq_lit(e, x, complex<6, 2>{1.0, 2.0}));
    e.add_unit(satx::eq_lit(e, y, complex<6, 2>{1.0, 3.0}));
    e.add_unit(satx::lt_lit(e, y, x));  // contradicción
    auto m = satx::solver::solve(e);
    assert(!m.has_value() && m.error() == result::unsat);
  }

  // pow simbólico: x = 1.5 → x³ == 3.25 (truncado, ±2^−F) y x^(−1) == 0.75
  // (división simbólica exacta-en-cuadrícula: trunc(r·x) == one) (§10.7)
  {
    satx::engine e;
    const complex<6, 2> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<6, 2>{1.5, 0.0}));
    const auto p = satx::pow(x, 3);
    const auto q = satx::pow(x, -1);
    // Identidad de la división simbólica: trunc(q·x) == 1.0 exacto.
    const auto back = q * x;
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert(close(p.value(*m), {3.375, 0.0}, 1.0 / 4.0));
    assert(close(q.value(*m), {2.0 / 3.0, 0.0}, 1.0 / 4.0));
    assert(close(back.value(*m), {1.0, 0.0}, 1e-9));
  }

  // root_cbe simbólico: y² == 4 → (y·y) == 4 bajo el modelo (§10.7)
  {
    satx::engine e;
    const auto y = satx::root_cbe(complex<6, 2>{4.0, 0.0}, 2);
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    const auto ys = y * y;
    assert(close(ys.value(*m), {4.0, 0.0}, 1.0 / 4.0));
  }

  // ── ADR-011/012/013: regresiones de la revisión del núcleo ──

  // Identidad x + 0 == x para W impar (la caja NB ⊄ rango con signo de W bits)
  {
    satx::engine e;
    const complex<3, 0> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<3, 0>{4, 0}));
    const auto z = x + complex<3, 0>::zero(e);
    const auto w = x + complex<3, 0>{1, 0};
    e.add_unit(satx::eq_lit(e, w, complex<3, 0>{5, 0}));  // 5 = max_NB(3)
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert((z.value(*m) == std::complex<double>{4.0, 0.0}));
    assert((w.value(*m) == std::complex<double>{5.0, 0.0}));
  }

  // Wrap simbólico de add coincide con el concreto (85 + 85 = −86 en W=8)
  {
    satx::engine e;
    const complex<8, 0> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<8, 0>{85, 0}));
    const auto s = x + x;
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert((s.value(*m) == std::complex<double>{-86.0, 0.0}));
  }

  // ADR-013: alineación de escala exacta en eq_lit (antes era vacua: x == 10
  // «satisfacía» x == 0 con F−Fa = W)
  {
    satx::engine e;
    const complex<8, 0> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<8, 0>{10, 0}));
    e.add_unit(satx::eq_lit(e, x, complex<8, 8>{0.0, 0.0}));  // 10 != 0 exacto
    auto m = satx::solver::solve(e);
    assert(!m.has_value() && m.error() == result::unsat);
  }
  {
    satx::engine e;
    const complex<8, 0> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<8, 0>{20, 0}));
    e.add_unit(satx::eq_lit(e, x, complex<8, 4>{4.0, 0.0}));  // 20 != 4 (shift parcial)
    auto m = satx::solver::solve(e);
    assert(!m.has_value() && m.error() == result::unsat);
  }

  // mul simbólico trunca hacia cero (ADR-012): (−0.5)·(0.0625) == 0
  {
    satx::engine e;
    const complex<8, 4> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<8, 4>{-0.5, 0.0}));
    const auto p = x * complex<8, 4>{0.0625, 0.0};
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert((p.value(*m) == std::complex<double>{0.0, 0.0}));
  }

  // División con divisor de escala mixta (regresión: antes UNSAT espurio por
  // construir la restricción de divisor no nulo con rieles re-escalados)
  {
    satx::engine e;
    const complex<4, 3> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<4, 3>{4, 0}));  // 0.5
    const auto q = x / complex<4, 1>{4, 0};               // / 2.0 → 0.25
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert(close(q.value(*m), {0.25, 0.0}, 1e-9));
  }

  // nb_exact_lit: barrido de patrones de W=3 (signado(p) ∈ [−2, 5])
  {
    for (int p = 0; p < 8; ++p) {
      satx::engine e;
      const complex<3, 0> x{e};
      e.add_unit(satx::eq_lit(e, x, complex<3, 0>::from_raw_wrap(p, 0)));
      const auto tc = x.tc_real_rail(e);
      const auto ex = satx::num::nb_exact_lit(e, tc);
      const bool want = (p < 4) || (p >= 6);
      if (want) e.add_unit(ex);
      else e.add_unit(-ex);
      auto m = satx::solver::solve(e);
      assert(m.has_value());
    }
  }

  // real()/imag() simbólicos (también como operandos de restricciones)
  {
    satx::engine e;
    const complex<8, 2> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<8, 2>{1.5, -2.0}));
    const auto r = satx::real(x);
    const auto im = satx::imag(x);
    e.add_unit(satx::eq_lit(e, r, complex<8, 2>{1.5, 0.0}));
    e.add_unit(satx::eq_lit(e, im, complex<8, 2>{-2.0, 0.0}));
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert((r.value(*m) == std::complex<double>{1.5, 0.0}));
    assert((im.value(*m) == std::complex<double>{-2.0, 0.0}));
  }

  // ADR-011: add/sub de ancho mixto con operandos negativos — la extensión del
  // patrón re-ancla con −2^Wa (antes: complex<4,0>{-10} + complex<8,0>{0} == 6)
  {
    satx::engine e;
    const complex<4, 0> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<4, 0>{-10, 0}));
    const auto s = x + complex<8, 0>{0, 0};
    const auto t = x + complex<8, 0>{10, 0};
    const auto u = x + complex<8, 0>{85, 0};
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert((s.value(*m) == std::complex<double>{-10.0, 0.0}));
    assert((t.value(*m) == std::complex<double>{0.0, 0.0}));
    assert((u.value(*m) == std::complex<double>{75.0, 0.0}));
  }
  {
    satx::engine e;
    const complex<4, 0> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<4, 0>{-10, 0}));
    const auto s = complex<8, 0>{85, 0} - x;  // 85 − (−10) = 95 → wrap → −161 (caja NB)
    auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert((s.value(*m) == std::complex<double>{-161.0, 0.0}));
  }

  // ADR-011: eq_lit/lt_lit/le_lit de tipos mixtos (anchos y escalas distintos)
  {
    satx::engine e;
    const complex<8, 0> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<8, 0>{10, 0}));
    e.add_unit(-satx::eq_lit(e, x, complex<8, 8>{0.0, 0.0}));  // 10 != 0
    auto m = satx::solver::solve(e);
    assert(m.has_value());
  }
  {
    satx::engine e;
    const complex<4, 0> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<4, 0>{-10, 0}));
    e.add_unit(satx::le_lit(e, x, complex<8, 0>{5, 0}));   // −10 <= 5
    e.add_unit(-satx::lt_lit(e, complex<8, 0>{5, 0}, x));  // ¬(5 < −10)
    auto m = satx::solver::solve(e);
    assert(m.has_value());
  }
  {
    satx::engine e;
    const complex<8, 4> x{e};
    e.add_unit(satx::eq_lit(e, x, complex<8, 4>{64, 0}));  // 4.0
    e.add_unit(satx::eq_lit(e, x, complex<4, 0>{4, 0}));   // 4.0 (mismo valor)
    auto m = satx::solver::solve(e);
    assert(m.has_value());
  }

  std::puts("test_kerberos: OK");
  return 0;
}
