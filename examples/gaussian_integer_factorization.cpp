// gaussian_integer_factorization — factorización de enteros gaussianos por SAT.
//
// Busca x, y ∈ Z[i] (no unidades) con x·y == n usando la ruta simbólica de
// satx: la multiplicación se compila a un circuito CBE y el kernel CDCL SLIME
// de Kerberos encuentra los factores. Con F=0 la grilla es Z[i] EXACTO
// (K = min(Fa,Fb) = 0, el producto no trunca) y solo los divisores gaussianos
// satisfacen la restricción.
//
//   5 = (1 + 2i)(1 − 2i) = (2 + i)(2 − i)
//
// Se enumeran todas las factorizaciones (módulo unidades y orden) con
// cláusulas de bloqueo y cada una se verifica con aritmética concreta.

#include <satx/satx.hpp>

#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
  using C = satx::complex<8, 0>;  // grilla entera: Z[i] ∩ [−170, 85]²

  const C n{5, 0};  // entero gaussiano a factorizar (raw 5)

  satx::engine e;
  const C x{e}, y{e};  // factores desconocidos

  // x · y == n
  e.add_unit(satx::eq_lit(e, x * y, n));
  // |x|² > 1 y |y|² > 1 (excluye unidades y el cero)
  e.add_unit(satx::lt_lit(e, C::one(e), satx::abs_sq(x)));
  e.add_unit(satx::lt_lit(e, C::one(e), satx::abs_sq(y)));
  // Romper la simetría x ↔ y (orden lexicográfico)
  e.add_unit(satx::le_lit(e, x, y));

  std::cout << "Factores de " << n.value({}) << " en Z[i] (x <= y):\n";

  int count = 0;
  while (count < 64) {
    const auto m = satx::solver::solve(e);
    if (!m) break;
    ++count;

    const std::complex<double> vx = x.value(*m);
    const std::complex<double> vy = y.value(*m);
    std::cout << "  " << vx << "  ×  " << vy << '\n';

    // Verificación independiente con la ruta concreta (aritmética exacta).
    const C cx{vx.real(), vx.imag()};
    const C cy{vy.real(), vy.imag()};
    if (!(cx * cy == n)) {
      std::cerr << "VERIFICACIÓN FALLIDA\n";
      return EXIT_FAILURE;
    }

    // Bloquear este modelo: cláusula ¬(x = vx ∧ y = vy).
    std::vector<satx::lit_t> block;
    const auto xr = x.re_pattern(), xi = x.im_pattern();
    const auto yr = y.re_pattern(), yi = y.im_pattern();
    for (std::size_t i = 0; i < xr.size(); ++i) {
      block.push_back(m->get(xr[i]) ? -xr[i] : xr[i]);
      block.push_back(m->get(xi[i]) ? -xi[i] : xi[i]);
      block.push_back(m->get(yr[i]) ? -yr[i] : yr[i]);
      block.push_back(m->get(yi[i]) ? -yi[i] : yi[i]);
    }
    e.add_clause(block);
  }

  std::cout << "Total: " << count << " factorizaciones (módulo unidades).\n";
  return count > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
