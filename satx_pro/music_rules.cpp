// music_rules — catálogo 5.5: composición musical con reglas, kernel SLIME +
// BASILISK.
//
// Melodía de 8 pulsos sobre la escala pentatónica {0, 2, 4, 7, 9}; reglas:
//   · una nota por pulso;
//   · sin saltos > 4 semitonos entre pulsos consecutivos;
//   · forma ABA: los pulsos 6-7 repiten 0-1.
// BASILISK cuenta los fragmentos de 2 pulsos con notas distintas (20).

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int NP = 8;  // pulsos
    const int scale[5] = {0, 2, 4, 7, 9};  // pentatónica mayor
    constexpr int NS = 5;

    satx::engine e;
    // n[t][p]: nota p en el pulso t
    std::vector<std::vector<satx::lit_t>> n(NP, std::vector<satx::lit_t>(NS));
    for (int t = 0; t < NP; ++t)
        for (int p = 0; p < NS; ++p) n[t][p] = e.add_variable();

    // una nota por pulso
    for (int t = 0; t < NP; ++t) {
        e.add_clause({n[t][0], n[t][1], n[t][2], n[t][3], n[t][4]});
        for (int p1 = 0; p1 < NS; ++p1)
            for (int p2 = p1 + 1; p2 < NS; ++p2)
                e.add_unit(-satx::gates::and2(e, n[t][p1], n[t][p2]));
    }
    // sin saltos > 4 semitonos
    for (int t = 0; t + 1 < NP; ++t)
        for (int p1 = 0; p1 < NS; ++p1)
            for (int p2 = 0; p2 < NS; ++p2)
                if (std::abs(scale[p1] - scale[p2]) > 4)
                    e.add_unit(-satx::gates::and2(e, n[t][p1], n[t + 1][p2]));
    // forma ABA: pulsos 6-7 = pulsos 0-1
    for (int p = 0; p < NS; ++p) {
        e.add_unit(-satx::gates::and2(e, n[6][p], -n[0][p]));
        e.add_unit(-satx::gates::and2(e, -n[6][p], n[0][p]));
        e.add_unit(-satx::gates::and2(e, n[7][p], -n[1][p]));
        e.add_unit(-satx::gates::and2(e, -n[7][p], n[1][p]));
    }

    std::printf("música: %d pulsos, escala pentatónica (variables %zu, cláusulas %zu)\n", NP,
                e.variable_count(), e.clause_count());

    const auto sol = satx::solver::solve(e);
    if (!sol) {
        std::printf("VERIFICACIÓN FALLIDA: se esperaba SAT\n");
        return EXIT_FAILURE;
    }

    std::printf("melodía:");
    std::vector<int> melody(NP);
    for (int t = 0; t < NP; ++t)
        for (int p = 0; p < NS; ++p)
            if (sol->get(n[t][p])) {
                melody[t] = scale[p];
                std::printf(" %d", scale[p]);
            }
    std::printf("\n");

    // ── verificación en el host ─────────────────────────────────────────────
    for (int t = 0; t + 1 < NP; ++t)
        if (std::abs(melody[t] - melody[t + 1]) > 4) {
            std::printf("VERIFICACIÓN FALLIDA: salto en pulso %d\n", t);
            return EXIT_FAILURE;
        }
    if (melody[6] != melody[0] || melody[7] != melody[1]) {
        std::printf("VERIFICACIÓN FALLIDA: forma ABA\n");
        return EXIT_FAILURE;
    }
    for (int t = 0; t < NP; ++t) {
        bool in_scale = false;
        for (int p = 0; p < NS; ++p) in_scale |= melody[t] == scale[p];
        if (!in_scale) {
            std::printf("VERIFICACIÓN FALLIDA: nota fuera de escala\n");
            return EXIT_FAILURE;
        }
    }
    std::printf("verificación del host: OK\n");

    // ── conteo: fragmentos de 2 pulsos con notas distintas ──────────────────
    {
        satx::engine e2;
        std::array<satx::lit_t, NS> a, b;
        for (int p = 0; p < NS; ++p) {
            a[p] = e2.add_variable();
            b[p] = e2.add_variable();
        }
        e2.add_clause({a[0], a[1], a[2], a[3], a[4]});
        e2.add_clause({b[0], b[1], b[2], b[3], b[4]});
        for (int p1 = 0; p1 < NS; ++p1)
            for (int p2 = p1 + 1; p2 < NS; ++p2) {
                e2.add_unit(-satx::gates::and2(e2, a[p1], a[p2]));
                e2.add_unit(-satx::gates::and2(e2, b[p1], b[p2]));
            }
        for (int p = 0; p < NS; ++p) e2.add_unit(-satx::gates::and2(e2, a[p], b[p]));
        const auto cnt = satx::solver::basilisk::count(e2);
        std::printf("fragmentos de 2 pulsos con notas distintas (BASILISK) = %s\n",
                    cnt.value().c_str());
        if (cnt.as_double() != 20.0) return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
