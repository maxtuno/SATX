// ═══════════════════════════════════════════════════════════════════════════
// template.cpp — «Hola, mundo» de satx.
//
// El flujo esencial del sistema, en siete líneas:
//   1. Elige un tipo de número complejo CBE: complex<W, F>.
//   2. Crea un motor (satx::engine) que compila las restricciones a un
//      circuito booleano (CNF).
//   3. Declara incógnitas complejas simbólicas.
//   4. Impone una restricción: z == x + y.
//   5. Pide al kernel CDCL SLIME de Kerberos que resuelva el circuito.
//   6. Si hay solución, lee los valores concretos de las incógnitas.
//
// Compilar y ejecutar:
//   cmake -S . -B build && cmake --build build
//   ./bin/template
// ═══════════════════════════════════════════════════════════════════════════

#include <satx/satx.hpp>
#include <cstdlib>
#include <iostream>

int main() {
    // complex<W, F>: números complejos CBE(W,F) — Complejo Binario
    // Entrelazado (formato original de Oscar Riveros, 2026). Cada carril
    // (real e imaginario) usa W bits en base −2 con F posiciones
    // fraccionarias. Aquí: 16 bits por carril y 4 bits fraccionarios.
    using C = satx::complex<16, 4>;

    // El motor acumula las restricciones y las compila a un circuito CNF.
    satx::engine e;

    // Incógnitas simbólicas: cada una es una variable booleana libre.
    const C x{e};
    const C y{e};
    const C z{e};

    // Restricción: z == x + y (el circuito de la suma se genera aquí).
    e.add_unit(satx::eq_lit(e, x + y, z));

    // Resolver con el kernel CDCL SLIME de Kerberos (embebido, sin
    // dependencias externas).
    auto s = satx::solver::solve(e);

    if (s) {
        std::cout << "x = " << x.value(*s) << '\n';
        std::cout << "y = " << y.value(*s) << '\n';
        std::cout << "z = " << z.value(*s) << '\n';
        std::cout << "x + y = " << (x.value(*s) + y.value(*s)) << '\n';
        std::cout << "z == x + y  ->  " << (z.value(*s) == x.value(*s) + y.value(*s) ? "true" : "false") << '\n';
    } else {
        std::cout << "Sin solución (el circuito es insatisfacible).\n";
    }

    return EXIT_SUCCESS;
}
