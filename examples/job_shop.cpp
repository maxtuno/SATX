#include <satx/satx.hpp>
#include <cstdlib>
#include <iostream>

// Problema: planificación (job-shop) con dos máquinas y fecha límite.
//   - Máquina 1: tarea A (2 h) y tarea B (1 h), sin solaparse.
//   - Máquina 2: tarea C (3 h) y tarea D (1 h), sin solaparse.
//   - Toda tarea empieza en t >= 0 y termina antes de la hora 6.
// Cada "no solapamiento" es una disyunción de precedencias; el solver elige
// el orden de ejecución en cada máquina.

int main() {
    using C = satx::complex<8, 0>;   // horas enteras (F = 0)

    satx::engine e;
    const C sA{e}, sB{e}, sC{e}, sD{e};   // hora de inicio de cada tarea

    const auto num = [](double v) { return C{v, 0.0}; };

    // horas reales: parte imaginaria a 0 y dominio {0..6} (así las sumas con
    // duraciones no envuelven el datapath TC de 8 bits)
    for (const auto* s : {&sA, &sB, &sC, &sD}) {
        for (const auto l : s->imag_rail(e)) e.add_unit(-l);
        satx::lit_t in06 = satx::eq_lit(e, *s, num(0));
        for (int v = 1; v <= 6; ++v)
            in06 = satx::gates::or2(e, in06, satx::eq_lit(e, *s, num(v)));
        e.add_unit(in06);
    }

    // máquina 1: A antes que B, o B antes que A
    e.add_unit(satx::gates::or2(e, satx::le_lit(e, sA + num(2), sB),
                                   satx::le_lit(e, sB + num(1), sA)));

    // máquina 2: C antes que D, o D antes que C
    e.add_unit(satx::gates::or2(e, satx::le_lit(e, sC + num(3), sD),
                                   satx::le_lit(e, sD + num(1), sC)));

    // fecha límite: hora 6
    e.add_unit(satx::le_lit(e, sA + num(2), num(6)));
    e.add_unit(satx::le_lit(e, sB + num(1), num(6)));
    e.add_unit(satx::le_lit(e, sC + num(3), num(6)));
    e.add_unit(satx::le_lit(e, sD + num(1), num(6)));

    if (auto sol = satx::solver::solve(e); sol) {
        std::cout << "maquina 1: A en " << sA.value(*sol).real()
                  << " h, B en " << sB.value(*sol).real() << " h\n";
        std::cout << "maquina 2: C en " << sC.value(*sol).real()
                  << " h, D en " << sD.value(*sol).real() << " h\n";
    }
    return EXIT_SUCCESS;
}
