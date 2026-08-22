#include <satx/satx.hpp>
#include <cstdlib>
#include <iostream>

// Problema: física cuántica — entrelazamiento (estado de Bell).
// Aplicar H al qubit 0 y CNOT(0 -> 1) al estado |00> produce el estado
// entrelazado (|00> + |11>)/sqrt(2). Ruta concreta: amplitudes CBE de punto
// fijo, sin solver; se verifica la normalización y el valor esperado <Z0> = 0.

int main() {
    using namespace satx::quantum;
    constexpr std::size_t W = 24, F = 12;
    const satx::solver::model m{};       // modelo vacío: solo decodifica

    qstate<W, F> s{2};                   // |00>
    s.apply1(0, h2<W, F>());             // (|0> + |1>)|0> / sqrt(2)
    s.apply2(0, 1, cnot4<W, F>());       // (|00> + |11>) / sqrt(2)

    const auto amps = s.decode(m);
    std::cout << "|00>: " << amps[0] << '\n';
    std::cout << "|01>: " << amps[1] << '\n';
    std::cout << "|10>: " << amps[2] << '\n';
    std::cout << "|11>: " << amps[3] << '\n';
    std::cout << "suma |psi|^2 = " << s.norm_sq_concrete() << '\n';
    std::cout << "<Z0> = " << s.expect_z(0).value(m).real() << '\n';
    return EXIT_SUCCESS;
}
