// complex_polynomial_roots — raíces de un polinomio con coeficientes CBE
// encontradas por SAT y enumeradas con cláusulas de bloqueo.
//
//   p(z) = z³ − i·z² − z + i = (z − 1)(z − i)(z + 1)
//
// La restricción p(x) == 0 usa pow (square & multiply de muls), add/sub e
// eq_lit; el kernel CDCL SLIME de Kerberos devuelve las raíces de la grilla
// CBE(8,2), y cada raíz se verifica con la evaluación concreta del polinomio.

#include <satx/satx.hpp>

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
  using C = satx::complex<8, 2>;

  satx::engine e;
  const C x{e};  // raíz desconocida

  // p(x) = x³ − i·x² − x + i == 0
  const C i = C::i_unit(e);
  const auto p = satx::pow(x, 3) - i * satx::pow(x, 2) - x + i;
  e.add_unit(satx::eq_lit(e, p, C::zero(e)));

  std::cout << "Raíces de p(z) = z³ − i·z² − z + i en CBE(8,2):\n";

  int count = 0;
  while (true) {
    const auto m = satx::solver::solve(e);
    if (!m) break;
    ++count;

    const std::complex<double> r = x.value(*m);
    std::cout << "  z = " << r << '\n';

    // Verificación independiente: evaluación concreta del polinomio.
    const C cr{r.real(), r.imag()};
    const std::complex<double> pv = (satx::pow(cr, 3) - i * satx::pow(cr, 2) - cr + i)
                                        .value({});
    if (std::abs(pv) > 1.0 / 4.0) {
      std::cerr << "  VERIFICACIÓN FALLIDA (p = " << pv << ")\n";
      return EXIT_FAILURE;
    }

    // Bloquear este modelo.
    std::vector<satx::lit_t> block;
    const auto xr = x.re_pattern(), xi = x.im_pattern();
    for (std::size_t k = 0; k < xr.size(); ++k) {
      block.push_back(m->get(xr[k]) ? -xr[k] : xr[k]);
      block.push_back(m->get(xi[k]) ? -xi[k] : xi[k]);
    }
    e.add_clause(block);
  }

  std::cout << "Total: " << count << " raíces (esperado 3: 1, −1, i)\n";
  return count == 3 ? EXIT_SUCCESS : EXIT_FAILURE;
}
