#include <satx/satx.hpp>
#include <cstdlib>
#include <complex>
#include <iostream>

// Problema: física cuántica — problema inverso (aprendizaje de B por SAT).
// Dado un dato OTOC C(2) medido con la compuerta desconocida B = X y el
// circuito U = [RX(0.7)] sobre un qubit, encontrar B sabiendo solo el dato.
// B se modela como una compuerta libre (coeficientes simbólicos) restringida
// a coeficientes reales y unitaria (constrain_unitary2), y se impone que el
// bucle del eco reproduzca el dato. El kernel SAT reconstruye B.

int main() {
    using namespace satx::quantum;
    constexpr std::size_t W = 8, F = 4;

    qcircuit<W, F> U;
    U.push(0, rx2<W, F>(0.7));

    // 1) dato: OTOC con B = X fijo, en el mismo instrumento numérico
    satx::engine e0;
    qstate<W, F> ref{e0, 1};
    const auto c2_true = otoc_echo<W, F>(ref, U, 0, x2<W, F>(), 0, id2<W, F>(), 1);
    const auto s0 = satx::solver::solve(e0);
    if (!s0) return EXIT_FAILURE;
    const std::complex<double> target = c2_true.value(*s0);

    // 2) aprendizaje: B libre y real, unitaria, con OTOC == dato
    satx::engine e;
    qstate<W, F> psi{e, 1};
    const auto B = free_gate2<W, F>(e);
    for (std::size_t i = 0; i < 4; ++i)
        for (const auto l : B.m[i].imag_rail(e)) e.add_unit(-l);   // B real
    constrain_unitary2(e, B);                                      // B en O(2)
    const auto c2 = otoc_echo<W, F>(psi, U, 0, B, 0, id2<W, F>(), 1);
    e.add_unit(satx::eq_lit(e, c2, cx<W, F>{target.real(), target.imag()}));

    if (auto sol = satx::solver::solve(e); sol) {
        std::cout << "dato OTOC C(2) = " << target << '\n';
        std::cout << "B aprendida:\n";
        for (const auto& b : B.m)
            std::cout << "  " << b.value(*sol) << '\n';

        // verificacion: la B aprendida (fija) debe reproducir el dato
        const auto b00 = B.m[0].value(*sol);
        const auto b01 = B.m[1].value(*sol);
        const auto b10 = B.m[2].value(*sol);
        const auto b11 = B.m[3].value(*sol);
        const auto B_rec = qgate2<W, F>{{cx<W, F>{b00.real(), b00.imag()},
                                          cx<W, F>{b01.real(), b01.imag()},
                                          cx<W, F>{b10.real(), b10.imag()},
                                          cx<W, F>{b11.real(), b11.imag()}}};
        satx::engine e2;
        qstate<W, F> chk{e2, 1};
        const auto chk_otoc = otoc_echo<W, F>(chk, U, 0, B_rec, 0, id2<W, F>(), 1);
        if (auto s2 = satx::solver::solve(e2); s2)
            std::cout << "verificacion: C(2) con B aprendida = "
                      << chk_otoc.value(*s2) << '\n';
    }
    return EXIT_SUCCESS;
}
