// dock_scheduling — catálogo 1.4: programación de muelles de carga, kernel
// SLIME (scheduling disyuntivo con aritmética CBE).
//
// Camiones con ventana de llegada [a_t, b_t], duración de descarga p_t y
// asignación a un muelle. Restricciones:
//   · ventanas:           a_t ≤ s_t ≤ b_t, s_t + p_t ≤ horizonte
//   · sin solapamiento:   por muelle, los intervalos de dos camiones no se
//                         superponen: (s_a + p_a ≤ s_b) ∨ (s_b + p_b ≤ s_a)
//                         ∨ (muelles distintos)
//   · incompatibilidad:   pares refrigerado/peligroso no se solapan jamás
//
// Verificación: comprobación independiente de ventanas, muelles y
// solapamientos en el host.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

struct truck {
    int a, b;          // ventana [a, b]
    int p;             // duración
    int type;          // 0 normal, 1 refrigerado, 2 peligroso
};

int main() {
    const std::vector<truck> trucks = {
        {0, 6, 3, 1},    // t0 refrigerado
        {2, 8, 2, 0},    // t1 normal
        {4, 12, 4, 2},   // t2 peligroso
        {6, 14, 2, 0},   // t3 normal
        {8, 16, 3, 1},   // t4 refrigerado
        {10, 20, 3, 2},  // t5 peligroso
    };
    constexpr int N = 6, D = 3;  // camiones, muelles
    constexpr int H = 24;        // horizonte

    using I = satx::integer<6>;  // tiempo 0..27 (rango -32..31)

    satx::engine e;
    std::vector<I> s;  // instante de inicio
    std::vector<I> d_; // muelle (0..2)
    for (int t = 0; t < N; ++t) {
        s.emplace_back(e);
        d_.emplace_back(e);
        // ventana
        e.add_unit(satx::le_lit(e, I{trucks[t].a}, s[t]));
        e.add_unit(satx::le_lit(e, s[t], I{trucks[t].b}));
        // fin dentro del horizonte
        e.add_unit(satx::le_lit(e, s[t] + I{trucks[t].p}, I{H}));
        // muelle válido
        e.add_unit(satx::le_lit(e, I{0}, d_[t]));
        e.add_unit(satx::le_lit(e, d_[t], I{D - 1}));
    }

    const auto precedes = [&](int a, int b) {
        return satx::le_lit(e, s[a] + I{trucks[a].p}, s[b]);
    };

    for (int a = 0; a < N; ++a)
        for (int b = a + 1; b < N; ++b) {
            // mismo muelle → sin solapamiento (disyunción)
            e.add_clause({precedes(a, b), precedes(b, a),
                          satx::ne_lit(e, d_[a], d_[b])});
            // incompatibilidad refrigerado/peligroso: jamás se solapan
            const bool cold_haz = (trucks[a].type == 1 && trucks[b].type == 2) ||
                                  (trucks[a].type == 2 && trucks[b].type == 1);
            if (cold_haz) e.add_clause({precedes(a, b), precedes(b, a)});
        }

    std::printf("muelles de carga: %d camiones, %d muelles, variables %zu, cláusulas %zu\n",
                N, D, e.variable_count(), e.clause_count());

    const auto sol = satx::solver::solve(e);
    if (!sol) {
        std::printf("VERIFICACIÓN FALLIDA: se esperaba SAT\n");
        return EXIT_FAILURE;
    }

    for (int t = 0; t < N; ++t)
        std::printf("  camión %d (tipo %d, p=%d): muelle %d en t=%d..%d\n", t, trucks[t].type,
                    trucks[t].p, static_cast<int>(d_[t].value(*sol)),
                    static_cast<int>(s[t].value(*sol)),
                    static_cast<int>(s[t].value(*sol)) + trucks[t].p);

    // ── verificación en el host ─────────────────────────────────────────────
    for (int t = 0; t < N; ++t) {
        const int st = static_cast<int>(s[t].value(*sol));
        if (st < trucks[t].a || st > trucks[t].b || st + trucks[t].p > H) {
            std::printf("VERIFICACIÓN FALLIDA: ventana del camión %d\n", t);
            return EXIT_FAILURE;
        }
    }
    for (int a = 0; a < N; ++a)
        for (int b = a + 1; b < N; ++b) {
            const int sa = static_cast<int>(s[a].value(*sol));
            const int sb = static_cast<int>(s[b].value(*sol));
            const int da = static_cast<int>(d_[a].value(*sol));
            const int db = static_cast<int>(d_[b].value(*sol));
            const bool overlap = sa < sb + trucks[b].p && sb < sa + trucks[a].p;
            if (overlap && da == db) {
                std::printf("VERIFICACIÓN FALLIDA: solapamiento en muelle (%d, %d)\n", a, b);
                return EXIT_FAILURE;
            }
            const bool cold_haz = (trucks[a].type == 1 && trucks[b].type == 2) ||
                                  (trucks[a].type == 2 && trucks[b].type == 1);
            if (cold_haz && overlap) {
                std::printf("VERIFICACIÓN FALLIDA: incompatibles (%d, %d)\n", a, b);
                return EXIT_FAILURE;
            }
        }
    std::printf("verificación del host: OK\n");
    return EXIT_SUCCESS;
}
