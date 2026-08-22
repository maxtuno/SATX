#include <satx/satx.hpp>
#include <cstdlib>
#include <iostream>

// Problema: física — tiro parabólico.
// ¿Con qué velocidad inicial v hay que lanzar un proyectil a 45° para alcanzar
// un blanco a 10 m, con g = 10 m/s²?
//   R == v² · sin(2θ) / g      con sin(90°) = 1  ->  v² == 100  ->  v == 10
// La velocidad es la incógnita; la ecuación (multiplicación y división
// simbólicas, en punto fijo de 8 bits fraccionarios) se compila a CNF y el
// solver encuentra v.

int main() {
    using C = satx::complex<16, 8>;   // punto fijo: 8 bits fraccionarios

    satx::engine e;
    const C v{e};                     // velocidad inicial (incógnita)

    const auto num = [](double x) { return C{x, 0.0}; };

    // velocidad real: parte imaginaria fijada a 0
    for (const auto l : v.imag_rail(e)) e.add_unit(-l);

    // dominio: 0 <= v <= 20 m/s
    e.add_unit(satx::le_lit(e, num(0), v));
    e.add_unit(satx::le_lit(e, v, num(20)));

    // R == (v² · sin(2θ)) / g, con θ = 45°, g = 10 y R = 10
    const auto range = (v * v * num(1.0)) / num(10.0);
    e.add_unit(satx::eq_lit(e, range, num(10.0)));

    if (auto sol = satx::solver::solve(e); sol) {
        const double v_found = v.value(*sol).real();
        std::cout << "velocidad inicial: " << v_found << " m/s\n";
        std::cout << "comprobacion: v^2/g = " << v_found * v_found / 10.0 << " m\n";
    }
    return EXIT_SUCCESS;
}
