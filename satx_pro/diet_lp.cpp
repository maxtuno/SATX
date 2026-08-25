// diet_lp — catálogo 4.1: dieta / mezcla de alimentos (LP clásico), kernel
// PIXIE.
//
//   min  Σ_i c_i·x_i
//   s.a. L_n ≤ Σ_i a_{n,i}·x_i ≤ U_n   ∀ nutriente
//        x_i ≥ 0
//
// Verificación: (a) las restricciones se cumplen en el host; (b) el óptimo
// LP es cota inferior de la búsqueda en rejilla del host.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace satx::solver::pixie;

int main() {
    constexpr int F = 4, N = 3;
    const double cost[F] = {2.0, 1.5, 3.0, 1.0};
    // nutrientes: cal, proteína, vitaminas
    const double a[N][F] = {
        {50.0, 20.0, 40.0, 30.0},
        {2.0, 1.0, 4.0, 1.0},
        {1.0, 3.0, 2.0, 1.0},
    };
    const double lo[N] = {300.0, 20.0, 15.0};
    const double hi[N] = {500.0, 1e9, 1e9};

    model m{"dieta"};
    std::vector<variable*> x;
    for (int i = 0; i < F; ++i) x.push_back(&m.add_continuous("x" + std::to_string(i), 0.0, 50.0));

    for (int n = 0; n < N; ++n) {
        expr e;
        for (int i = 0; i < F; ++i) e += a[n][i] * (*x[i]);
        m.add_constraint(e, compare::ge, lo[n]);
        if (hi[n] < 1e8) m.add_constraint(e, compare::le, hi[n]);
    }
    expr obj;
    for (int i = 0; i < F; ++i) obj += cost[i] * (*x[i]);
    m.set_objective(obj, sense::min);

    const auto s = m.solve();

    std::printf("estado: %s\n",
                s.state() == status::optimal ? "OPTIMAL"
                : s.state() == status::infeasible ? "INFEASIBLE" : "UNKNOWN");
    if (s.state() != status::optimal || !s.has_values()) return EXIT_FAILURE;

    std::printf("costo mínimo (pixie LP) = %.6f\n", s.objective());
    for (int i = 0; i < F; ++i) std::printf("  alimento %d: %.4f\n", i, s.value(*x[i]));

    // ── verificación (a): restricciones ─────────────────────────────────────
    for (int n = 0; n < N; ++n) {
        double v = 0.0;
        for (int i = 0; i < F; ++i) v += a[n][i] * s.value(*x[i]);
        std::printf("  nutriente %d: %.4f (≥ %g%s)\n", n, v, lo[n],
                    v + 1e-6 >= lo[n] ? " ok" : " FAIL");
        if (v + 1e-6 < lo[n]) return EXIT_FAILURE;
    }

    // ── verificación (b): cota inferior por rejilla ─────────────────────────
    double grid = 1e18;
    for (double x0 = 0.0; x0 <= 12.0; x0 += 0.5)
        for (double x1 = 0.0; x1 <= 12.0; x1 += 0.5)
            for (double x2 = 0.0; x2 <= 12.0; x2 += 0.5)
                for (double x3 = 0.0; x3 <= 12.0; x3 += 0.5) {
                    const double vv[N] = {
                        a[0][0] * x0 + a[0][1] * x1 + a[0][2] * x2 + a[0][3] * x3,
                        a[1][0] * x0 + a[1][1] * x1 + a[1][2] * x2 + a[1][3] * x3,
                        a[2][0] * x0 + a[2][1] * x1 + a[2][2] * x2 + a[2][3] * x3,
                    };
                    if (vv[0] < lo[0] || vv[0] > hi[0] || vv[1] < lo[1] || vv[2] < lo[2]) continue;
                    grid = std::min(grid, cost[0] * x0 + cost[1] * x1 + cost[2] * x2 + cost[3] * x3);
                }
    std::printf("costo mínimo en rejilla (host) = %.2f\n", grid);
    if (s.objective() > grid + 1e-4) {
        std::printf("VERIFICACIÓN FALLIDA: el LP supera la rejilla\n");
        return EXIT_FAILURE;
    }
    std::printf("verificación del host: OK\n");
    return EXIT_SUCCESS;
}
