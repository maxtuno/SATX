// menu_planning — catálogo 5.10: planificación de menús con nutrición y
// presupuesto, kernel WMIBO.
//
// 4 días × 2 comidas; 6 platos con costo, alergeno y puntaje nutricional.
// Restricciones: un plato por comida, exclusión del plato con frutos secos
// (quedan 5 platos utilizables), presupuesto semanal, variedad (cada plato
// ≤ 3 veces). Objetivo: maximizar nutrición; cláusula blanda: preferencia
// por que el plato 4 aparezca al menos una vez.
//
// Verificación: recomputación de presupuesto, variedad, alergias y objetivo
// (nutrición − penalización blanda) en el host.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace satx::solver::wmibo;

int main() {
    constexpr int DAYS = 4, MEALS = 2, DISHES = 6;
    const int cost[DISHES] = {2, 3, 4, 3, 4, 5};
    const int nutr[DISHES] = {5, 4, 3, 6, 4, 2};
    const bool allergen[DISHES] = {0, 0, 1, 0, 0, 0};  // solo el plato 2 con frutos secos
    // Presupuesto 26: con variedad ≤ 3 y 8 comidas, el mínimo es
    // 3·2 + 3·3 + 2·3 = 21 (platos 0, 1, 3).
    constexpr int BUDGET = 26;

    model m{"menu"};

    // x[slot][dish], slot = día*MEALS + comida
    std::vector<std::vector<variable*>> x(DAYS * MEALS, std::vector<variable*>(DISHES));
    for (int slot = 0; slot < DAYS * MEALS; ++slot)
        for (int d = 0; d < DISHES; ++d) x[slot][d] = &m.add_boolean("x" + std::to_string(slot) + "_" + std::to_string(d));

    // un plato por comida
    for (int slot = 0; slot < DAYS * MEALS; ++slot) {
        expr e;
        for (int d = 0; d < DISHES; ++d) e += *x[slot][d];
        m.add_constraint(e, compare::eq, 1.0);
    }
    // alergias
    for (int slot = 0; slot < DAYS * MEALS; ++slot)
        for (int d = 0; d < DISHES; ++d)
            if (allergen[d]) m.add_hard_clause({~lit{*x[slot][d]}});
    // presupuesto
    expr budget;
    for (int slot = 0; slot < DAYS * MEALS; ++slot)
        for (int d = 0; d < DISHES; ++d) budget += static_cast<double>(cost[d]) * (*x[slot][d]);
    m.add_constraint(budget, compare::le, BUDGET);
    // variedad: cada plato a lo sumo 3 veces
    for (int d = 0; d < DISHES; ++d)
        for (int s1 = 0; s1 < DAYS * MEALS; ++s1)
            for (int s2 = s1 + 1; s2 < DAYS * MEALS; ++s2)
                for (int s3 = s2 + 1; s3 < DAYS * MEALS; ++s3)
                    for (int s4 = s3 + 1; s4 < DAYS * MEALS; ++s4)
                        m.add_hard_clause({~lit{*x[s1][d]}, ~lit{*x[s2][d]}, ~lit{*x[s3][d]},
                                           ~lit{*x[s4][d]}});
    // objetivo: nutrición
    expr obj;
    for (int slot = 0; slot < DAYS * MEALS; ++slot)
        for (int d = 0; d < DISHES; ++d) obj += static_cast<double>(nutr[d]) * (*x[slot][d]);
    m.set_objective(obj, sense::max);
    // blanda: el plato 4 (índice 3) aparece al menos una vez (peso 3)
    {
        std::vector<lit> lits;
        for (int slot = 0; slot < DAYS * MEALS; ++slot) lits.emplace_back(*x[slot][3]);
        m.add_soft_clause(3.0, lits);
    }

    const auto s = m.solve();

    std::printf("estado: %s\n",
                s.state() == status::optimal ? "OPTIMAL"
                : s.state() == status::infeasible ? "INFEASIBLE" : "UNKNOWN");
    if (s.state() != status::optimal || !s.has_values()) return EXIT_FAILURE;

    std::printf("objetivo óptimo (wmibo) = %.0f (nutrición − penalizaciones)\n", s.objective());

    // ── verificación en el host ─────────────────────────────────────────────
    int spend = 0, nutrition = 0, count4 = 0;
    for (int day = 0; day < DAYS; ++day) {
        std::printf("  día %d:", day);
        for (int meal = 0; meal < MEALS; ++meal) {
            const int slot = day * MEALS + meal;
            int chosen = -1, cnt = 0;
            for (int d = 0; d < DISHES; ++d)
                if (s.boolean(*x[slot][d])) { chosen = d; ++cnt; }
            if (cnt != 1) {
                std::printf("\nVERIFICACIÓN FALLIDA: comida %d del día %d\n", meal, day);
                return EXIT_FAILURE;
            }
            if (allergen[chosen]) {
                std::printf("\nVERIFICACIÓN FALLIDA: alérgeno en día %d\n", day);
                return EXIT_FAILURE;
            }
            spend += cost[chosen];
            nutrition += nutr[chosen];
            if (chosen == 3) ++count4;
            std::printf("  plato%d", chosen);
        }
        std::printf("\n");
    }
    std::printf("gasto = %d/%d, variedad: plato4 x%d\n", spend, BUDGET, count4);
    if (spend > BUDGET) {
        std::printf("VERIFICACIÓN FALLIDA: presupuesto\n");
        return EXIT_FAILURE;
    }
    std::vector<int> freq(DISHES, 0);
    for (int slot = 0; slot < DAYS * MEALS; ++slot)
        for (int d = 0; d < DISHES; ++d)
            if (s.boolean(*x[slot][d])) ++freq[d];
    for (int d = 0; d < DISHES; ++d)
        if (freq[d] > 3) {
            std::printf("VERIFICACIÓN FALLIDA: plato %d repetido %d veces\n", d, freq[d]);
            return EXIT_FAILURE;
        }
    const double verified = static_cast<double>(nutrition) - (count4 == 0 ? 3.0 : 0.0);
    std::printf("objetivo verificado = %.0f %s\n", verified,
                std::abs(verified - s.objective()) < 1e-6 ? "(ok)" : "(FAIL)");
    if (std::abs(verified - s.objective()) >= 1e-6) return EXIT_FAILURE;

    // ── caso infactible: presupuesto 5 ──────────────────────────────────────
    {
        model m2{"menu_tight"};
        std::vector<std::vector<variable*>> y(DAYS * MEALS, std::vector<variable*>(DISHES));
        for (int slot = 0; slot < DAYS * MEALS; ++slot)
            for (int d = 0; d < DISHES; ++d) y[slot][d] = &m2.add_boolean("y" + std::to_string(slot) + "_" + std::to_string(d));
        for (int slot = 0; slot < DAYS * MEALS; ++slot) {
            expr e;
            for (int d = 0; d < DISHES; ++d) e += *y[slot][d];
            m2.add_constraint(e, compare::eq, 1.0);
        }
        expr budget2;
        for (int slot = 0; slot < DAYS * MEALS; ++slot)
            for (int d = 0; d < DISHES; ++d) budget2 += static_cast<double>(cost[d]) * (*y[slot][d]);
        m2.add_constraint(budget2, compare::le, 5.0);
        const auto s2 = m2.solve();
        std::printf("presupuesto 5 → %s (esperado INFEASIBLE)\n",
                    s2.state() == status::infeasible ? "INFEASIBLE" : "otro");
        if (s2.state() != status::infeasible) return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
