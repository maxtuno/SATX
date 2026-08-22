#include <satx/satx.hpp>
#include <cstdlib>
#include <array>
#include <iostream>

// Problema: distribución de probabilidad de un dado trucado.
// Encontrar p1..p6 en [0, 1] (fracciones de resolución 2^-3) tales que:
//   p1 + ... + p6 == 1
//   esperanza E[X] == 4.0, con E = 1·p1 + 2·p2 + ... + 6·p6
//   p3 == p5
// El solver devuelve una distribución válida (p. ej. p3 = p5 = 0.5).

int main() {
    // F = 3 (resolución 1/8): los coeficientes 1..6 y las probabilidades deben
    // caber en el rango negabinario de W = 8, cuyo máximo es max_NB(8) = 85
    // (6·2^3 = 48 <= 85; con F = 4, 6·2^4 = 96 quedaría fuera de rango).
    using C = satx::complex<8, 3>;   // probabilidades: 3 bits fraccionarios

    satx::engine e;
    const C p1{e}, p2{e}, p3{e}, p4{e}, p5{e}, p6{e};
    const std::array p = {p1, p2, p3, p4, p5, p6};

    const auto num = [](double v) { return C{v, 0.0}; };

    // probabilidades reales: parte imaginaria fijada a 0
    for (const auto& pi : p)
        for (const auto l : pi.imag_rail(e)) e.add_unit(-l);

    // dominio [0, 1]
    for (const auto& pi : p) {
        e.add_unit(satx::le_lit(e, num(0), pi));
        e.add_unit(satx::le_lit(e, pi, num(1)));
    }

    // suma == 1
    const auto total = p1 + p2 + p3 + p4 + p5 + p6;
    e.add_unit(satx::eq_lit(e, total, num(1.0)));

    // esperanza == 4.0
    const auto mean = p1 * num(1) + p2 * num(2) + p3 * num(3)
                    + p4 * num(4) + p5 * num(5) + p6 * num(6);
    e.add_unit(satx::eq_lit(e, mean, num(4.0)));

    // p3 == p5
    e.add_unit(satx::eq_lit(e, p3, p5));

    if (auto sol = satx::solver::solve(e); sol) {
        double acc = 0.0, e_acc = 0.0;
        for (std::size_t i = 0; i < p.size(); ++i) {
            const double pi = p[i].value(*sol).real();
            acc += pi;
            e_acc += (i + 1) * pi;
            std::cout << "p" << (i + 1) << " = " << pi << '\n';
        }
        std::cout << "suma = " << acc << ", esperanza = " << e_acc << '\n';
    }
    return EXIT_SUCCESS;
}
