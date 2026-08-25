// lockers — catálogo 2.3: asignación de casilleros y puntos de recogida,
// kernel SLIME + BASILISK (#SAT).
//
// Paquetes → casilleros con capacidad 1 y compatibilidad térmica (los
// paquetes fríos solo pueden ir a casilleros fríos). Se cuenta con BASILISK
// el número de configuraciones admisibles para dimensionar la red, y se
// comprueba el caso infactible (menos casilleros que paquetes: palomar).

#include <satx/satx.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int NP = 4, NL = 4;
    // 0,1 = fríos (solo casilleros 0,1); 2,3 = ambiente
    const int cold[NP] = {1, 1, 0, 0};

    satx::engine e;
    std::vector<std::vector<satx::lit_t>> x(NP, std::vector<satx::lit_t>(NL));
    for (int p = 0; p < NP; ++p)
        for (int l = 0; l < NL; ++l) x[p][l] = e.add_variable();

    // cada paquete en exactamente un casillero
    for (int p = 0; p < NP; ++p) {
        e.add_clause({x[p][0], x[p][1], x[p][2], x[p][3]});
        for (int l1 = 0; l1 < NL; ++l1)
            for (int l2 = l1 + 1; l2 < NL; ++l2)
                e.add_unit(-satx::gates::and2(e, x[p][l1], x[p][l2]));
    }
    // capacidad 1 por casillero (AMO)
    for (int l = 0; l < NL; ++l)
        for (int p1 = 0; p1 < NP; ++p1)
            for (int p2 = p1 + 1; p2 < NP; ++p2)
                e.add_unit(-satx::gates::and2(e, x[p1][l], x[p2][l]));
    // compatibilidad: frío solo en casilleros fríos (0, 1)
    for (int p = 0; p < NP; ++p)
        if (cold[p])
            for (int l = 2; l < NL; ++l) e.add_unit(-x[p][l]);

    std::printf("casilleros: %d paquetes, %d casilleros (variables %zu, cláusulas %zu)\n",
                NP, NL, e.variable_count(), e.clause_count());

    // ── decisión + conteo ───────────────────────────────────────────────────
    const auto sol = satx::solver::solve(e);
    if (!sol) {
        std::printf("VERIFICACIÓN FALLIDA: se esperaba SAT\n");
        return EXIT_FAILURE;
    }
    std::printf("asignación: ");
    for (int p = 0; p < NP; ++p)
        for (int l = 0; l < NL; ++l)
            if (sol->get(x[p][l])) std::printf("p%d→L%d ", p, l);
    std::printf("\n");

    const auto cnt = satx::solver::basilisk::count(e);
    std::printf("configuraciones admisibles (BASILISK) = %s\n", cnt.value().c_str());

    // ── oráculo del host: permutaciones biyectivas válidas ──────────────────
    int oracle = 0;
    std::vector<int> a = {0, 1, 2, 3};
    do {
        bool ok = true;
        for (int p = 0; p < NP && ok; ++p)
            if (cold[p] && a[p] >= 2) ok = false;
        if (ok) ++oracle;
    } while (std::next_permutation(a.begin(), a.end()));

    std::printf("configuraciones admisibles (oráculo) = %d %s\n", oracle,
                cnt.as_double() == static_cast<double>(oracle) ? "(ok)" : "(FAIL)");
    if (cnt.as_double() != static_cast<double>(oracle)) return EXIT_FAILURE;

    // ── caso infactible: 3 casilleros para 4 paquetes (palomar) ────────────
    {
        satx::engine e2;
        std::vector<std::vector<satx::lit_t>> y(NP, std::vector<satx::lit_t>(3));
        for (int p = 0; p < NP; ++p)
            for (int l = 0; l < 3; ++l) y[p][l] = e2.add_variable();
        for (int p = 0; p < NP; ++p) {
            e2.add_clause({y[p][0], y[p][1], y[p][2]});
            for (int l1 = 0; l1 < 3; ++l1)
                for (int l2 = l1 + 1; l2 < 3; ++l2)
                    e2.add_unit(-satx::gates::and2(e2, y[p][l1], y[p][l2]));
        }
        for (int l = 0; l < 3; ++l)
            for (int p1 = 0; p1 < NP; ++p1)
                for (int p2 = p1 + 1; p2 < NP; ++p2)
                    e2.add_unit(-satx::gates::and2(e2, y[p1][l], y[p2][l]));
        const auto sol2 = satx::solver::solve(e2);
        std::printf("4 paquetes / 3 casilleros → %s (esperado UNSAT)\n", sol2 ? "SAT" : "UNSAT");
        if (sol2) return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
