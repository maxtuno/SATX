// test_solver_api — opciones, estadísticas, suposiciones y sesiones (ADR-011)
// Cubre las extensiones del puente Kerberos: SAT bajo suposiciones, reúso
// incremental del handle (cláusulas aprendidas) y conteo de modelos por
// cláusulas de bloqueo.

#include <satx/satx.hpp>

#include <cassert>
#include <cstdio>
#include <vector>

using satx::solver::result;

int main() {
  // Suposiciones: F = (a ∨ b); ¬a → SAT con b; {¬a, ¬b} → UNSAT; {a} → SAT
  {
    satx::engine e;
    const satx::lit_t a = e.add_variable();
    const satx::lit_t b = e.add_variable();
    e.add_clause({a, b});

    auto m1 = satx::solver::solve(e, {-a});
    assert(m1.has_value() && m1->get(b) == true);
    auto m2 = satx::solver::solve(e, {-a, -b});
    assert(!m2.has_value() && m2.error() == result::unsat);
    auto m3 = satx::solver::solve(e, {a});
    assert(m3.has_value() && m3->get(a) == true);
    assert(satx::solver::solve(e).has_value());  // sin suposiciones sigue SAT
  }

  // Suposiciones inválidas → std::invalid_argument
  {
    satx::engine e;
    e.add_variable();
    bool threw = false;
    try {
      (void)satx::solver::solve(e, {0});
    } catch (std::invalid_argument const&) {
      threw = true;
    }
    assert(threw);
    threw = false;
    try {
      (void)satx::solver::solve(e, {7});
    } catch (std::invalid_argument const&) {
      threw = true;
    }
    assert(threw);
  }

  // Opciones del kernel + estadísticas
  {
    satx::engine e;
    const satx::lit_t a = e.add_variable();
    const satx::lit_t b = e.add_variable();
    e.add_clause({a, b});  // binaria: aparece en binary_clauses de slime
    satx::solver::options opt;
    opt.heuristic_mode = 1;  // CHB
    satx::solver::stats st;
    auto m = satx::solver::solve(e, opt, &st);
    assert(m.has_value());
    assert(st.clauses > 0);
    assert(st.conflicts >= 0 && st.propagations >= 0 && st.decisions >= 0);
  }

  // Sesión incremental: reúso del handle entre llamadas con suposiciones
  {
    satx::engine e;
    const satx::lit_t a = e.add_variable();
    const satx::lit_t b = e.add_variable();
    e.add_clause({a, b});

    satx::solver::session s{e};
    assert(s.solve().has_value());
    auto m1 = s.solve({-a});  // fuerza b
    assert(m1.has_value() && m1->get(b) == true);
    auto m2 = s.solve({-a, -b});
    assert(!m2.has_value() && m2.error() == result::unsat);
    assert(s.solve().has_value());  // las suposiciones se revierten tras la llamada
    assert(s.variable_count() == e.variable_count());
  }

  // Conteo de modelos por cláusulas de bloqueo (a ∨ b) → 3 modelos
  {
    satx::engine e;
    const satx::lit_t a = e.add_variable();
    const satx::lit_t b = e.add_variable();
    e.add_clause({a, b});

    int count = 0;
    while (true) {
      auto m = satx::solver::solve(e);
      if (!m) break;
      ++count;
      std::vector<satx::lit_t> block;
      block.push_back(m->get(a) ? -a : a);
      block.push_back(m->get(b) ? -b : b);
      e.add_clause(block);  // ¬(modelo): excluye solo este modelo
    }
    assert(count == 3);
  }

  // Conteo CBE por bloqueo: |x|² == 1 en W=4, F=0 → x ∈ {1, −1, i, −i} (4 modelos)
  {
    using C = satx::complex<4, 0>;
    satx::engine e;
    const C x{e};
    e.add_unit(satx::eq_lit(e, satx::abs_sq(x), C::one(e)));

    int count = 0;
    while (true) {
      auto m = satx::solver::solve(e);
      if (!m) break;
      ++count;
      const double re = x.value(*m).real();
      const double im = x.value(*m).imag();
      assert(std::abs(std::hypot(re, im) - 1.0) < 1e-9);  // unidad exacta en la grilla
      std::vector<satx::lit_t> block;
      const auto xr = x.re_pattern();
      const auto xi = x.im_pattern();
      for (std::size_t i = 0; i < xr.size(); ++i) {
        block.push_back(m->get(xr[i]) ? -xr[i] : xr[i]);
        block.push_back(m->get(xi[i]) ? -xi[i] : xi[i]);
      }
      e.add_clause(block);  // ¬(modelo)
    }
    assert(count == 4);
  }

  std::puts("test_solver_api: OK");
  return 0;
}
