// maintenance — catálogo 3.4: planificación de mantenimiento preventivo,
// kernel SLIME (time-indexed).
//
// Máquinas con periodicidad máxima P (cada ventana de P+1 semanas debe
// contener una intervención) y cuadrillas limitadas por semana. Se resuelve
// una instancia factible y una infactible (P=2 con una sola cuadrilla:
// 4×4 > 12 slots → UNSAT).

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int M = 4, H = 12;

    const auto build = [](satx::engine& e, int P, int crews,
                          std::vector<std::vector<satx::lit_t>>& x) {
        for (int m = 0; m < M; ++m)
            for (int t = 0; t < H; ++t) x[m][t] = e.add_variable();

        // periodicidad: Σ_{t'=t}^{t+P} x ≥ 1
        for (int m = 0; m < M; ++m)
            for (int t = 0; t + P < H; ++t) {
                std::vector<satx::lit_t> c;
                for (int t2 = t; t2 <= t + P; ++t2) c.push_back(x[m][t2]);
                e.add_clause(c);
            }
        // recursos: a lo sumo `crews` intervenciones por semana
        if (crews == 1) {
            for (int t = 0; t < H; ++t)
                for (int m1 = 0; m1 < M; ++m1)
                    for (int m2 = m1 + 1; m2 < M; ++m2)
                        e.add_unit(-satx::gates::and2(e, x[m1][t], x[m2][t]));
        } else {
            for (int t = 0; t < H; ++t)
                for (int m1 = 0; m1 < M; ++m1)
                    for (int m2 = m1 + 1; m2 < M; ++m2)
                        for (int m3 = m2 + 1; m3 < M; ++m3) {
                            const satx::lit_t a = satx::gates::and2(e, x[m1][t], x[m2][t]);
                            e.add_unit(-satx::gates::and2(e, a, x[m3][t]));
                        }
        }
    };

    // ── (a) factible: P=3, 2 cuadrillas ─────────────────────────────────────
    {
        constexpr int P = 3;
        satx::engine e;
        std::vector<std::vector<satx::lit_t>> x(M, std::vector<satx::lit_t>(H));
        build(e, P, 2, x);
        std::printf("(a) P=3, 2 cuadrillas: variables %zu, cláusulas %zu\n", e.variable_count(),
                    e.clause_count());
        const auto sol = satx::solver::solve(e);
        if (!sol) {
            std::printf("VERIFICACIÓN FALLIDA: se esperaba SAT\n");
            return EXIT_FAILURE;
        }
        // verificación: cada ventana de P+1 semanas debe contener ≥ 1
        // intervención ⟺ primera ≤ P, huecos ≤ P+1, última ≥ H−1−P
        for (int m = 0; m < M; ++m) {
            int first = -1, last = -1, count = 0;
            for (int t = 0; t < H; ++t)
                if (sol->get(x[m][t])) {
                    if (last >= 0 && t - last > P + 1) {
                        std::printf("VERIFICACIÓN FALLIDA: máquina %d, hueco > %d\n", m, P + 1);
                        return EXIT_FAILURE;
                    }
                    if (first < 0) first = t;
                    last = t;
                    ++count;
                }
            if (first < 0 || first > P || last < H - 1 - P) {
                std::printf("VERIFICACIÓN FALLIDA: máquina %d sin periodicidad\n", m);
                return EXIT_FAILURE;
            }
            std::printf("  máquina %d: %d intervenciones\n", m, count);
        }
        for (int t = 0; t < H; ++t) {
            int c = 0;
            for (int m = 0; m < M; ++m) c += sol->get(x[m][t]) ? 1 : 0;
            if (c > 2) {
                std::printf("VERIFICACIÓN FALLIDA: cuadrillas en semana %d\n", t);
                return EXIT_FAILURE;
            }
        }
        std::printf("verificación del host: OK\n");
    }

    // ── (b) infactible: P=2, 1 cuadrilla → UNSAT ────────────────────────────
    {
        satx::engine e;
        std::vector<std::vector<satx::lit_t>> x(M, std::vector<satx::lit_t>(H));
        build(e, 2, 1, x);
        const auto sol = satx::solver::solve(e);
        std::printf("(b) P=2, 1 cuadrilla → %s (esperado UNSAT)\n", sol ? "SAT" : "UNSAT");
        if (sol) return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
