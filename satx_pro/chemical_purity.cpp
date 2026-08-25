// chemical_purity — catálogo 4.4: mezclas de productos químicos con pureza y
// reactividad, kernel SLIME.
//
// Componentes con pureza p_i, costo c_i y cantidades discretizadas x_i;
// y_i indica presencia (y_i ↔ x_i ≥ 1). Restricciones:
//   · balance de masa:      Σ x_i = 12
//   · pureza:               Σ p_i·x_i ≥ 96   (≥ 80 %)
//   · incompatibilidades:   ¬y_0 ∨ ¬y_2, ¬y_1 ∨ ¬y_3
//   · presupuesto:          Σ c_i·x_i ≤ B
//
// Se busca el presupuesto mínimo con búsqueda binaria (reconstruyendo la
// fórmula por B) y se compara contra el oráculo del host (11^5 = 161051
// combinaciones).

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int C = 5;
    const int purity[C] = {9, 8, 7, 6, 5};
    const int cost[C] = {4, 3, 2, 2, 1};
    constexpr int MASS = 12, PURE_MIN = 96;

    using I = satx::integer<8>;  // cantidades 0..10 y cotas 0..96 (rango -128..127)

    const auto build = [&](satx::engine& e, int B, std::vector<I>& x) {
        std::vector<satx::lit_t> y;
        for (int i = 0; i < C; ++i) {
            x.emplace_back(e);
            y.push_back(e.add_variable());
            e.add_unit(satx::le_lit(e, I{0}, x[i]));
            e.add_unit(satx::le_lit(e, x[i], I{10}));
            // y_i ↔ x_i ≥ 1
            e.add_clause({-y[i], satx::ge_lit(e, x[i], I{1})});
            e.add_clause({y[i], satx::le_lit(e, x[i], I{0})});
        }
        // balance de masa
        e.add_unit(satx::eq_lit(e, x[0] + x[1] + x[2] + x[3] + x[4], I{MASS}));
        // pureza
        satx::lit_t pure = satx::ge_lit(e, I{purity[0]} * x[0] + I{purity[1]} * x[1] +
                                               I{purity[2]} * x[2] + I{purity[3]} * x[3] +
                                               I{purity[4]} * x[4],
                                           I{PURE_MIN});
        e.add_unit(pure);
        // incompatibilidades
        e.add_unit(-satx::gates::and2(e, y[0], y[2]));
        e.add_unit(-satx::gates::and2(e, y[1], y[3]));
        // presupuesto
        e.add_unit(satx::le_lit(e, I{cost[0]} * x[0] + I{cost[1]} * x[1] + I{cost[2]} * x[2] +
                                       I{cost[3]} * x[3] + I{cost[4]} * x[4],
                                   I{B}));
    };

    // búsqueda binaria del presupuesto mínimo
    int lo = 0, hi = 60;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        satx::engine e;
        std::vector<I> x;
        build(e, mid, x);
        if (satx::solver::solve(e)) hi = mid;
        else lo = mid + 1;
    }
    std::printf("presupuesto mínimo (SLIME) = %d\n", lo);

    // ── oráculo del host ────────────────────────────────────────────────────
    int oracle = 1 << 30;
    for (int x0 = 0; x0 <= 10; ++x0)
        for (int x1 = 0; x1 <= 10; ++x1)
            for (int x2 = 0; x2 <= 10; ++x2)
                for (int x3 = 0; x3 <= 10; ++x3) {
                    const int x4 = MASS - x0 - x1 - x2 - x3;
                    if (x4 < 0 || x4 > 10) continue;
                    const int pur = 9 * x0 + 8 * x1 + 7 * x2 + 6 * x3 + 5 * x4;
                    if (pur < PURE_MIN) continue;
                    if (x0 > 0 && x2 > 0) continue;
                    if (x1 > 0 && x3 > 0) continue;
                    const int cst = 4 * x0 + 3 * x1 + 2 * x2 + 2 * x3 + 1 * x4;
                    oracle = std::min(oracle, cst);
                }
    std::printf("presupuesto mínimo (oráculo) = %d %s\n", oracle, lo == oracle ? "(ok)" : "(FAIL)");
    if (lo != oracle) return EXIT_FAILURE;

    // plan con el presupuesto mínimo
    satx::engine e;
    std::vector<I> x;
    build(e, lo, x);
    const auto sol = satx::solver::solve(e);
    if (!sol) return EXIT_FAILURE;
    std::printf("lote: ");
    int pur = 0;
    for (int i = 0; i < C; ++i) {
        const int v = static_cast<int>(x[i].value(*sol));
        std::printf("x%d=%d ", i, v);
        pur += purity[i] * v;
    }
    std::printf("(pureza %d/120)\n", pur);
    return EXIT_SUCCESS;
}
