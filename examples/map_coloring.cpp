#include <satx/satx.hpp>
#include <cstdlib>
#include <array>
#include <iostream>
#include <utility>

// Problema: colorear un mapa (Australia sin Tasmania) con 3 colores de modo
// que dos estados vecinos nunca compartan color. Es coloración de grafos:
// cada estado es una variable con dominio {0, 1, 2} y cada frontera es una
// restricción de desigualdad.

int main() {
    using C = satx::complex<8, 0>;   // un color por estado: 0, 1 o 2

    satx::engine e;

    const C wa{e}, nt{e}, sa{e}, q{e}, nsw{e}, v{e};
    const std::array states = {wa, nt, sa, q, nsw, v};
    const std::array names = {"WA", "NT", "SA", "Q", "NSW", "V"};

    const auto num = [](double c) { return C{c, 0.0}; };

    // dominio: cada estado toma uno de los 3 colores
    for (const auto& s : states) {
        const auto c0 = satx::eq_lit(e, s, num(0));
        const auto c1 = satx::eq_lit(e, s, num(1));
        const auto c2 = satx::eq_lit(e, s, num(2));
        e.add_unit(satx::gates::or2(e, c0, satx::gates::or2(e, c1, c2)));
    }

    // fronteras: vecinos con colores distintos
    const std::array<std::pair<int, int>, 9> borders = {{
        {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}, {2, 4}, {2, 5}, {3, 4}, {4, 5},
    }};
    for (const auto [a, b] : borders)
        e.add_unit(-satx::eq_lit(e, states[a], states[b]));

    if (auto sol = satx::solver::solve(e); sol)
        for (std::size_t i = 0; i < states.size(); ++i)
            std::cout << names[i] << " -> color "
                      << static_cast<int>(states[i].value(*sol).real()) << '\n';
    return EXIT_SUCCESS;
}
