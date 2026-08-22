#include <satx/satx.hpp>
#include <cstdlib>
#include <array>
#include <iostream>

// Problema: las N reinas (N = 4). Colocar 4 reinas en un tablero 4x4 sin que
// se ataquen: una por fila, columnas distintas y sin diagonales compartidas.
// q[i] = columna de la reina de la fila i; las diagonales se excluyen con
// |q[i] - q[j]| != |i - j|.

int main() {
    constexpr int N = 4;
    using C = satx::complex<8, 0>;   // columnas: enteros 0..N-1

    satx::engine e;

    std::array<C, N> q;
    for (auto& qi : q) qi = C{e};

    const auto num = [](double v) { return C{v, 0.0}; };

    // dominio: columna válida
    for (const auto& qi : q) {
        satx::lit_t in_domain = satx::eq_lit(e, qi, num(0));
        for (int c = 1; c < N; ++c)
            in_domain = satx::gates::or2(e, in_domain, satx::eq_lit(e, qi, num(c)));
        e.add_unit(in_domain);
    }

    // columnas y diagonales libres
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            e.add_unit(-satx::eq_lit(e, q[i], q[j]));              // misma columna
            e.add_unit(-satx::eq_lit(e, q[i] - q[j], num(j - i))); // diagonal ascendente
            e.add_unit(-satx::eq_lit(e, q[i] - q[j], num(i - j))); // diagonal descendente
        }

    if (auto sol = satx::solver::solve(e); sol) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                std::cout << (static_cast<int>(q[i].value(*sol).real()) == j ? "Q " : ". ");
            std::cout << '\n';
        }
    }
    return EXIT_SUCCESS;
}
