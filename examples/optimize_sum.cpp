#include <satx/satx.hpp>
#include <cstdlib>
#include <iostream>

// Problema: optimización usando SAT como oráculo.
// Maximizar x + y con x, y enteros en [0, 10] y presupuesto x·y <= 20.
// La API no tiene rama de optimización: se itera pidiendo en cada vuelta una
// solución estrictamente mayor que la mejor encontrada, hasta que la fórmula
// se vuelve insatisfacible (ese es el óptimo).

int main() {
    using C = satx::complex<8, 0>;   // enteros (F = 0)

    satx::engine e;
    const C x{e}, y{e};

    const auto num = [](double v) { return C{v, 0.0}; };

    // solo enteros reales: parte imaginaria fijada a 0
    for (const auto l : x.imag_rail(e)) e.add_unit(-l);
    for (const auto l : y.imag_rail(e)) e.add_unit(-l);

    // dominio [0, 10]
    e.add_unit(satx::le_lit(e, num(0), x));
    e.add_unit(satx::le_lit(e, x, num(10)));
    e.add_unit(satx::le_lit(e, num(0), y));
    e.add_unit(satx::le_lit(e, y, num(10)));

    // presupuesto: x·y <= 20
    const auto product = x * y;
    e.add_unit(satx::le_lit(e, product, num(20)));

    const auto total = x + y;

    double best = -1.0, bx = 0.0, by = 0.0;
    for (;;) {
        const auto sol = satx::solver::solve(e);
        if (!sol) break;                    // sin solución mayor: óptimo alcanzado
        const double v = total.value(*sol).real();
        if (v > best) { best = v; bx = x.value(*sol).real(); by = y.value(*sol).real(); }
        e.add_unit(satx::lt_lit(e, num(best), total));   // la próxima debe ser > mejor
    }

    std::cout << "mejor solucion: x = " << bx << ", y = " << by
              << "  (x + y = " << best << ", x*y = " << bx * by << ")\n";
    return EXIT_SUCCESS;
}
