// sports_scheduling — catálogo 5.2: calendarios deportivos (round-robin con
// restricciones), kernel SLIME.
//
// Liga de n equipos, n−1 rondas: todos contra todos exactamente una vez, un
// partido por equipo y ronda, y a lo sumo dos partidos de local consecutivos.
// x[i][j][r]: en la ronda r, i recibe a j; h[i][r]: localía de i en r.
//
// Verificación: comprobación de round-robin y localía en el host.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int N = 6, R = N - 1;  // equipos, rondas

    satx::engine e;
    std::vector<std::vector<std::vector<satx::lit_t>>> x(
        N, std::vector<std::vector<satx::lit_t>>(N, std::vector<satx::lit_t>(R)));
    std::vector<std::vector<satx::lit_t>> h(N, std::vector<satx::lit_t>(R));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (i != j)
                for (int r = 0; r < R; ++r) x[i][j][r] = e.add_variable();
    for (int i = 0; i < N; ++i)
        for (int r = 0; r < R; ++r) h[i][r] = e.add_variable();

    // todos contra todos: para cada par {i,j}, exactamente una orientación y
    // ronda
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            std::vector<satx::lit_t> c;
            for (int r = 0; r < R; ++r) {
                c.push_back(x[i][j][r]);
                c.push_back(x[j][i][r]);
            }
            e.add_clause(c);  // al menos uno
            for (int a = 0; a < 2 * R; ++a)
                for (int b = a + 1; b < 2 * R; ++b) {
                    const satx::lit_t la = a < R ? x[i][j][a] : x[j][i][a - R];
                    const satx::lit_t lb = b < R ? x[i][j][b] : x[j][i][b - R];
                    e.add_unit(-satx::gates::and2(e, la, lb));
                }
        }

    // un partido por equipo y ronda
    for (int i = 0; i < N; ++i)
        for (int r = 0; r < R; ++r) {
            std::vector<satx::lit_t> c;
            for (int j = 0; j < N; ++j) {
                if (i != j) c.push_back(x[i][j][r]);
                if (i != j) c.push_back(x[j][i][r]);
            }
            e.add_clause(c);
            for (int a = 0; a < static_cast<int>(c.size()); ++a)
                for (int b = a + 1; b < static_cast<int>(c.size()); ++b)
                    e.add_unit(-satx::gates::and2(e, c[a], c[b]));
        }

    // localía: h[i][r] ↔ ∨_j x[i][j][r]
    for (int i = 0; i < N; ++i)
        for (int r = 0; r < R; ++r) {
            for (int j = 0; j < N; ++j)
                if (i != j) e.add_unit(-satx::gates::and2(e, x[i][j][r], -h[i][r]));
            std::vector<satx::lit_t> c;
            for (int j = 0; j < N; ++j)
                if (i != j) c.push_back(x[i][j][r]);
            c.push_back(-h[i][r]);
            e.add_clause(c);
        }

    // a lo sumo dos partidos de local consecutivos
    for (int i = 0; i < N; ++i)
        for (int r = 0; r + 2 < R; ++r) {
            const satx::lit_t a = satx::gates::and2(e, h[i][r], h[i][r + 1]);
            e.add_unit(-satx::gates::and2(e, a, h[i][r + 2]));
        }

    std::printf("calendario: %d equipos, %d rondas (variables %zu, cláusulas %zu)\n", N, R,
                e.variable_count(), e.clause_count());

    const auto sol = satx::solver::solve(e);
    if (!sol) {
        std::printf("VERIFICACIÓN FALLIDA: se esperaba SAT\n");
        return EXIT_FAILURE;
    }

    for (int r = 0; r < R; ++r) {
        std::printf("  ronda %d:", r);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (i != j && sol->get(x[i][j][r])) std::printf("  %d vs %d (en casa %d)", i, j, i);
        std::printf("\n");
    }

    // ── verificación en el host ─────────────────────────────────────────────
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            int cnt = 0;
            for (int r = 0; r < R; ++r) cnt += (sol->get(x[i][j][r]) || sol->get(x[j][i][r])) ? 1 : 0;
            if (cnt != 1) {
                std::printf("VERIFICACIÓN FALLIDA: par (%d, %d) juega %d veces\n", i, j, cnt);
                return EXIT_FAILURE;
            }
        }
    for (int i = 0; i < N; ++i)
        for (int r = 0; r < R; ++r) {
            int cnt = 0;
            for (int j = 0; j < N; ++j) {
                if (i != j && sol->get(x[i][j][r])) ++cnt;
                if (i != j && sol->get(x[j][i][r])) ++cnt;
            }
            if (cnt != 1) {
                std::printf("VERIFICACIÓN FALLIDA: equipo %d ronda %d (%d partidos)\n", i, r, cnt);
                return EXIT_FAILURE;
            }
        }
    for (int i = 0; i < N; ++i)
        for (int r = 0; r + 2 < R; ++r)
            if (sol->get(h[i][r]) && sol->get(h[i][r + 1]) && sol->get(h[i][r + 2])) {
                std::printf("VERIFICACIÓN FALLIDA: equipo %d, 3 locales seguidos\n", i);
                return EXIT_FAILURE;
            }
    std::printf("verificación del host: OK\n");
    return EXIT_SUCCESS;
}
