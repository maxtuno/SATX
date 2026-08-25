// nurse_rostering — catálogo 5.1: programación de personal con reglas
// legales (nurse rostering), kernel SLIME.
//
// Turnos D/E/N + descanso por enfermera y día; cobertura por (día, turno);
// reglas legales: un turno por día, no noche seguida de día, máximo de dos
// noches consecutivas. Se resuelve una instancia factible y una infactible
// (cobertura doble → 42 asignaciones > 21 disponibles).

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int NS = 3, ND = 7;  // enfermeras, días
    // req[d][s]: cobertura mínima por (día, turno)
    const int req1[ND][3] = {
        {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
    };

    const auto build = [](satx::engine& e, int const (&req)[ND][3],
                          std::vector<std::vector<std::array<satx::lit_t, 3>>>& x) {
        // x[n][d][s]: turno s (0=D, 1=E, 2=N); off[n][d]: descanso
        std::vector<std::vector<satx::lit_t>> off(NS, std::vector<satx::lit_t>(ND));
        for (int n = 0; n < NS; ++n)
            for (int d = 0; d < ND; ++d) {
                off[n][d] = e.add_variable();
                for (int s = 0; s < 3; ++s) x[n][d][s] = e.add_variable();
                // exactamente un turno (o descanso)
                e.add_clause({off[n][d], x[n][d][0], x[n][d][1], x[n][d][2]});
                for (int i = 0; i < 4; ++i)
                    for (int j = i + 1; j < 4; ++j) {
                        const satx::lit_t a = i == 3 ? off[n][d] : x[n][d][i];
                        const satx::lit_t b = j == 3 ? off[n][d] : x[n][d][j];
                        e.add_unit(-satx::gates::and2(e, a, b));
                    }
            }
        // cobertura
        for (int d = 0; d < ND; ++d)
            for (int s = 0; s < 3; ++s) {
                std::vector<satx::lit_t> c;
                for (int n = 0; n < NS; ++n) c.push_back(x[n][d][s]);
                if (req[d][s] == 1) e.add_clause(c);
                else {  // req = 2: al menos dos
                    for (int n1 = 0; n1 < NS; ++n1)
                        for (int n2 = n1 + 1; n2 < NS; ++n2) {
                            const satx::lit_t a = satx::gates::and2(e, x[n1][d][s], x[n2][d][s]);
                            for (int n3 = 0; n3 < NS; ++n3) e.add_clause({a, x[n3][d][s]});
                        }
                }
            }
        // reglas legales
        for (int n = 0; n < NS; ++n)
            for (int d = 0; d + 1 < ND; ++d) {
                // no N seguido de D
                e.add_unit(-satx::gates::and2(e, x[n][d][2], x[n][d + 1][0]));
                // máximo de dos noches consecutivas
                if (d + 2 < ND) {
                    const satx::lit_t a = satx::gates::and2(e, x[n][d][2], x[n][d + 1][2]);
                    e.add_unit(-satx::gates::and2(e, a, x[n][d + 2][2]));
                }
            }
    };

    // ── (a) factible ────────────────────────────────────────────────────────
    {
        satx::engine e;
        std::vector<std::vector<std::array<satx::lit_t, 3>>> x(
            NS, std::vector<std::array<satx::lit_t, 3>>(ND));
        build(e, req1, x);
        std::printf("(a) cobertura 1: variables %zu, cláusulas %zu\n", e.variable_count(),
                    e.clause_count());
        const auto sol = satx::solver::solve(e);
        if (!sol) {
            std::printf("VERIFICACIÓN FALLIDA: se esperaba SAT\n");
            return EXIT_FAILURE;
        }
        for (int d = 0; d < ND; ++d) {
            std::printf("  día %d:", d);
            for (int n = 0; n < NS; ++n) {
                int s = -1;
                for (int k = 0; k < 3; ++k)
                    if (sol->get(x[n][d][k])) s = k;
                std::printf("  N%d:%s", n, s == -1 ? "L" : (s == 0 ? "D" : s == 1 ? "E" : "N"));
            }
            std::printf("\n");
        }
        // verificación
        for (int n = 0; n < NS; ++n)
            for (int d = 0; d < ND; ++d) {
                int cnt = 0;
                for (int s = 0; s < 3; ++s) cnt += sol->get(x[n][d][s]) ? 1 : 0;
                if (cnt != 1) {
                    std::printf("VERIFICACIÓN FALLIDA: enfermera %d día %d\n", n, d);
                    return EXIT_FAILURE;
                }
            }
        for (int n = 0; n < NS; ++n)
            for (int d = 0; d + 1 < ND; ++d)
                if (sol->get(x[n][d][2]) && sol->get(x[n][d + 1][0])) {
                    std::printf("VERIFICACIÓN FALLIDA: N→D (%d, %d)\n", n, d);
                    return EXIT_FAILURE;
                }
        for (int n = 0; n < NS; ++n)
            for (int d = 0; d + 2 < ND; ++d)
                if (sol->get(x[n][d][2]) && sol->get(x[n][d + 1][2]) && sol->get(x[n][d + 2][2])) {
                    std::printf("VERIFICACIÓN FALLIDA: 3 noches (%d, %d)\n", n, d);
                    return EXIT_FAILURE;
                }
        for (int d = 0; d < ND; ++d)
            for (int s = 0; s < 3; ++s) {
                int cnt = 0;
                for (int n = 0; n < NS; ++n) cnt += sol->get(x[n][d][s]) ? 1 : 0;
                if (cnt < req1[d][s]) {
                    std::printf("VERIFICACIÓN FALLIDA: cobertura (%d, %d)\n", d, s);
                    return EXIT_FAILURE;
                }
            }
        std::printf("verificación del host: OK\n");
    }

    // ── (b) infactible: cobertura doble → UNSAT ─────────────────────────────
    {
        const int req2[ND][3] = {
            {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2},
        };
        satx::engine e;
        std::vector<std::vector<std::array<satx::lit_t, 3>>> x(
            NS, std::vector<std::array<satx::lit_t, 3>>(ND));
        build(e, req2, x);
        const auto sol = satx::solver::solve(e);
        std::printf("(b) cobertura doble → %s (esperado UNSAT)\n", sol ? "SAT" : "UNSAT");
        if (sol) return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
