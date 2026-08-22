// mandelbrot_escape — el conjunto de Mandelbrot como problema SAT.
//
// Ruta concreta: itera z ← z² + c en punto fijo y decide si c escapa
// (|z|² > 4) en K pasos.
//
// Ruta simbólica: busca un c en el rectángulo clásico [−2, 0.5] × [−1, 1]
// cuyo tiempo de escape sea EXACTAMENTE K: |z_K|² > 4 y |z_i|² ≤ 4 para
// i < K. El circuito de la órbita (muls + adds + abs_sq + lt_lit) se
// resuelve con el kernel CDCL SLIME de Kerberos (heurística CHB vía
// satx::solver::options — el orden VSIDS por defecto tarda ~25× más en
// esta instancia), y cada c encontrado se verifica con la simulación
// concreta.

#include <satx/satx.hpp>

#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr double FOUR = 4.0;

// Tiempo de escape de la órbita concreta (0 = nunca escapa en K pasos).
int escape_time(std::complex<double> c, int k) {
  std::complex<double> z{0.0, 0.0};
  for (int i = 1; i <= k; ++i) {
    z = z * z + c;
    if (std::norm(z) > FOUR) return i;
  }
  return 0;
}

}  // namespace

int main() {
  constexpr std::size_t K = 3;

  // ── 1. Órbitas concretas (constexpr-able, aritmética de punto fijo) ──
  {
    using C = satx::complex<16, 8>;
    const auto orbit = [](C c) {
      C z{0.0, 0.0};
      int t = 0;
      for (int i = 1; i <= static_cast<int>(K); ++i) {
        z = z * z + c;
        const double m2 = satx::abs_sq(z).value({}).real();
        if (m2 > FOUR) {
          t = i;
          break;
        }
      }
      return t;
    };
    std::cout << "Órbitas concretas (W=16, F=8):\n";
    for (auto c : {C{-0.8, 0.2}, C{0.3, 0.5}, C{0.1, 0.1}})
      std::cout << "  c = " << c.value({}) << "  → tiempo de escape "
                << orbit(c) << '\n';
  }

  // ── 2. Problema inverso por SAT: c con tiempo de escape exactamente K ──
  {
    using C = satx::complex<12, 5>;
    const C four{4.0, 0.0};

    satx::engine e;
    const C c{e};  // incógnita: el parámetro del mapa

    // Rectángulo clásico de la visualización.
    e.add_unit(satx::le_lit(e, C{-2.0, 0.0}, satx::real(c)));
    e.add_unit(satx::le_lit(e, satx::real(c), C{0.5, 0.0}));
    e.add_unit(satx::le_lit(e, C{-1.0, 0.0}, satx::imag(c)));
    e.add_unit(satx::le_lit(e, satx::imag(c), C{1.0, 0.0}));

    // Órbita: |z_i|² ≤ 4 (no escapa) para i = 1..K−1; |z_K|² > 4 (escapa).
    C z = c;
    e.add_unit(-satx::lt_lit(e, four, satx::abs_sq(z)));  // z_1 no escapa
    for (std::size_t i = 2; i < K; ++i) {
      z = z * z + c;
      e.add_unit(-satx::lt_lit(e, four, satx::abs_sq(z)));
    }
    z = z * z + c;  // z_K
    e.add_unit(satx::lt_lit(e, four, satx::abs_sq(z)));

    satx::solver::options opt;
    opt.heuristic_mode = 1;  // CHB

    std::cout << "\nBuscando c con tiempo de escape exactamente " << K << "...\n";
    int count = 0;
    while (count < 3) {
      const auto m = satx::solver::solve(e, opt);
      if (!m) break;
      ++count;

      const std::complex<double> vc = c.value(*m);
      std::cout << "  c = " << vc << '\n';

      // Verificación independiente en doble precisión.
      const int t = escape_time(vc, static_cast<int>(K));
      if (t != static_cast<int>(K)) {
        std::cerr << "VERIFICACIÓN FALLIDA (escape en " << t << " pasos)\n";
        return EXIT_FAILURE;
      }

      // Bloquear este modelo y seguir enumerando.
      std::vector<satx::lit_t> block;
      const auto cr = c.re_pattern(), ci = c.im_pattern();
      for (std::size_t i = 0; i < cr.size(); ++i) {
        block.push_back(m->get(cr[i]) ? -cr[i] : cr[i]);
        block.push_back(m->get(ci[i]) ? -ci[i] : ci[i]);
      }
      e.add_clause(block);
    }
    std::cout << "Encontrados: " << count << '\n';
  }

  return EXIT_SUCCESS;
}
