#include <satx/satx.hpp>
#include <cstdlib>
#include <array>
#include <iostream>

// Problema: criptoaritmética SEND + MORE = MONEY.
// Cada letra es un dígito (0..9), letras distintas tienen dígitos distintos,
// S y M no pueden ser 0, y la suma debe cumplirse columna a columna con
// acarreos c1..c4 en {0, 1}. El solver encuentra la asignación de dígitos.
//
// Columnas (de la menos significativa a la más significativa):
//   D + E      == Y + 10·c1
//   N + R + c1 == E + 10·c2
//   E + O + c2 == N + 10·c3
//   S + M + c3 == O + 10·c4
//   c4         == M

int main() {
    using C = satx::complex<8, 0>;   // dígitos y acarreos: enteros (F = 0)

    satx::engine e;

    const C s{e}, l_e{e}, n{e}, d{e};    // SEND
    const C m{e}, o{e}, r{e}, y{e};      // MORE / MONEY
    const C c1{e}, c2{e}, c3{e}, c4{e};  // acarreos

    const auto num = [](double v) { return C{v, 0.0}; };

    // dominio de los dígitos: 0..9
    const std::array letters = {s, l_e, n, d, m, o, r, y};
    for (const auto& l : letters) {
        satx::lit_t in09 = satx::eq_lit(e, l, num(0));
        for (int v = 1; v <= 9; ++v)
            in09 = satx::gates::or2(e, in09, satx::eq_lit(e, l, num(v)));
        e.add_unit(in09);
    }

    // letras distintas
    for (std::size_t i = 0; i < letters.size(); ++i)
        for (std::size_t j = i + 1; j < letters.size(); ++j)
            e.add_unit(-satx::eq_lit(e, letters[i], letters[j]));

    // acarreos en {0, 1}
    for (const auto& c : {c1, c2, c3, c4})
        e.add_unit(satx::gates::or2(e, satx::eq_lit(e, c, num(0)),
                                       satx::eq_lit(e, c, num(1))));

    // S y M no nulos
    e.add_unit(-satx::eq_lit(e, s, num(0)));
    e.add_unit(-satx::eq_lit(e, m, num(0)));

    // ecuaciones de columna
    e.add_unit(satx::eq_lit(e, d + l_e,          y + c1 * num(10)));
    e.add_unit(satx::eq_lit(e, n + r + c1,       l_e + c2 * num(10)));
    e.add_unit(satx::eq_lit(e, l_e + o + c2,     n + c3 * num(10)));
    e.add_unit(satx::eq_lit(e, s + m + c3,       o + c4 * num(10)));
    e.add_unit(satx::eq_lit(e, c4, m));

    if (auto sol = satx::solver::solve(e); sol) {
        std::cout << "SEND + MORE = MONEY\n";
        std::cout << "  S = " << s.value(*sol).real()       << '\n';
        std::cout << "  E = " << l_e.value(*sol).real()     << '\n';
        std::cout << "  N = " << n.value(*sol).real()       << '\n';
        std::cout << "  D = " << d.value(*sol).real()       << '\n';
        std::cout << "  M = " << m.value(*sol).real()       << '\n';
        std::cout << "  O = " << o.value(*sol).real()       << '\n';
        std::cout << "  R = " << r.value(*sol).real()       << '\n';
        std::cout << "  Y = " << y.value(*sol).real()       << '\n';
    }
    return EXIT_SUCCESS;
}
