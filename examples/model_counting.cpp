// model_counting — conteo de modelos (#SAT) por enumeración con cláusulas de
// bloqueo, sobre restricciones aritméticas CBE.
//
// (a) Sesión incremental del kernel Kerberos: reúso del handle CDCL entre
//     llamadas con suposiciones.
// (b) Idempotentes de la grilla: x·x == x en W=4 → {0, 1}.
// (c) Unidades complejas: |x|² == 1 en W=4 → {1, −1, i, −i}.
//
// El kernel BASILISK de Kerberos (backend #SAT exacto del CLI) cuenta los
// mismos problemas sin enumeración; aquí se muestra el patrón portable con
// el puente embebido.

#include <satx/satx.hpp>

#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
  // ── (a) Sesión incremental ──────────────────────────────────────────────
  {
    satx::engine e;
    const satx::lit_t a = e.add_variable();
    const satx::lit_t b = e.add_variable();
    e.add_clause({a, b});

    satx::solver::session s{e};
    std::cout << "(a) Sesión incremental sobre (a ∨ b):\n";
    std::cout << "    solve()        → " << (s.solve() ? "SAT" : "UNSAT") << '\n';
    std::cout << "    solve({¬a})    → " << (s.solve({-a}) ? "SAT (b=1)" : "UNSAT")
              << '\n';
    std::cout << "    solve({¬a,¬b}) → " << (s.solve({-a, -b}) ? "SAT" : "UNSAT")
              << '\n';
    std::cout << "    solve()        → " << (s.solve() ? "SAT (suposiciones revertidas)"
                                                      : "UNSAT")
              << '\n';
  }

  // ── (b) Idempotentes: x·x == x ──────────────────────────────────────────
  {
    using C = satx::complex<4, 0>;
    satx::engine e;
    const C x{e};
    e.add_unit(satx::eq_lit(e, x * x, x));

    std::cout << "\n(b) Idempotentes de la grilla CBE(4,0): x·x == x\n";
    int count = 0;
    while (true) {
      const auto m = satx::solver::solve(e);
      if (!m) break;
      ++count;
      std::cout << "    x = " << x.value(*m) << '\n';
      std::vector<satx::lit_t> block;
      for (auto l : x.re_pattern()) block.push_back(m->get(l) ? -l : l);
      for (auto l : x.im_pattern()) block.push_back(m->get(l) ? -l : l);
      e.add_clause(block);
    }
    std::cout << "    Total: " << count << " (esperado 2)\n";
    if (count != 2) return EXIT_FAILURE;
  }

  // ── (c) Unidades complejas: |x|² == 1 ───────────────────────────────────
  {
    using C = satx::complex<4, 0>;
    satx::engine e;
    const C x{e};
    e.add_unit(satx::eq_lit(e, satx::abs_sq(x), C::one(e)));

    std::cout << "\n(c) Unidades de la grilla CBE(4,0): |x|² == 1\n";
    int count = 0;
    while (true) {
      const auto m = satx::solver::solve(e);
      if (!m) break;
      ++count;
      std::cout << "    x = " << x.value(*m) << '\n';
      std::vector<satx::lit_t> block;
      const auto xr = x.re_pattern(), xi = x.im_pattern();
      for (std::size_t i = 0; i < xr.size(); ++i) {
        block.push_back(m->get(xr[i]) ? -xr[i] : xr[i]);
        block.push_back(m->get(xi[i]) ? -xi[i] : xi[i]);
      }
      e.add_clause(block);
    }
    std::cout << "    Total: " << count << " (esperado 4)\n";
    if (count != 4) return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
