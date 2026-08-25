// order_batching — catálogo 2.2: agrupación de pedidos en oleadas de picking,
// kernel WMIBO.
//
//   min  Σ_{i,b} w_i·b·y_{i,b}            (prioridades × oleada)
//   s.a. Σ_b y_{i,b} = 1                  (cada pedido en una oleada)
//        Σ_i it_i·y_{i,b} ≤ C             (capacidad de la oleada)
//        Σ_b b·y_{i,b} ≤ l_i              (plazo del pedido urgente)
//
// Verificación: oráculo del host — 3^6 asignaciones con filtrado de
// capacidad y plazos.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace satx::solver::wmibo;

int main() {
    constexpr int n = 6;  // pedidos
    constexpr int B = 3;  // oleadas
    constexpr int C = 6;  // capacidad (ítems) por oleada
    const int it[n] = {2, 3, 1, 4, 2, 3};
    const int w[n] = {1, 1, 3, 1, 2, 1};        // prioridad (más alta = antes)
    const int deadline[n] = {9, 9, 2, 9, 2, 9}; // oleada máxima (9 = no urgente)

    model m{"order_batching"};

    std::vector<std::vector<variable*>> y(n, std::vector<variable*>(B));
    for (int i = 0; i < n; ++i)
        for (int b = 0; b < B; ++b) y[i][b] = &m.add_boolean("y" + std::to_string(i) + "_" + std::to_string(b));

    // cada pedido en exactamente una oleada
    for (int i = 0; i < n; ++i) {
        expr e;
        for (int b = 0; b < B; ++b) e += *y[i][b];
        m.add_constraint(e, compare::eq, 1.0);
    }
    // capacidad de oleada
    for (int b = 0; b < B; ++b) {
        expr cap;
        for (int i = 0; i < n; ++i) cap += static_cast<double>(it[i]) * (*y[i][b]);
        m.add_constraint(cap, compare::le, C);
    }
    // plazos: Σ_b b·y_{i,b} ≤ deadline_i
    for (int i = 0; i < n; ++i) {
        expr d;
        for (int b = 0; b < B; ++b) d += static_cast<double>(b + 1) * (*y[i][b]);
        m.add_constraint(d, compare::le, static_cast<double>(deadline[i]));
    }
    // objetivo: Σ w_i·(b+1)·y_{i,b} (minimizar índice ponderado de oleada)
    expr obj;
    for (int i = 0; i < n; ++i)
        for (int b = 0; b < B; ++b) obj += static_cast<double>(w[i] * (b + 1)) * (*y[i][b]);
    m.set_objective(obj, sense::min);

    const auto s = m.solve();

    std::printf("estado: %s\n",
                s.state() == status::optimal ? "OPTIMAL"
                : s.state() == status::infeasible ? "INFEASIBLE" : "UNKNOWN");
    if (s.state() != status::optimal || !s.has_values()) return EXIT_FAILURE;

    std::printf("objetivo óptimo (wmibo) = %.0f\n", s.objective());
    for (int b = 0; b < B; ++b) {
        std::printf("  oleada %d:", b + 1);
        for (int i = 0; i < n; ++i)
            if (s.boolean(*y[i][b])) std::printf(" %d(it=%d,w=%d)", i, it[i], w[i]);
        std::printf("\n");
    }

    // ── oráculo del host ────────────────────────────────────────────────────
    double oracle = std::numeric_limits<double>::infinity();
    std::vector<int> a(n, 0);
    const std::size_t total = [] {
        std::size_t t = 1;
        for (int i = 0; i < n; ++i) t *= B;
        return t;
    }();
    for (std::size_t code = 0; code < total; ++code) {
        std::size_t k = code;
        for (int i = 0; i < n; ++i) { a[i] = static_cast<int>(k % B); k /= B; }
        std::vector<int> cap(B, 0);
        double obj = 0.0;
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            cap[a[i]] += it[i];
            if (cap[a[i]] > C) ok = false;
            if (a[i] + 1 > deadline[i]) ok = false;
            obj += w[i] * (a[i] + 1);
        }
        if (ok) oracle = std::min(oracle, obj);
    }

    std::printf("objetivo óptimo (oráculo) = %.0f %s\n", oracle,
                std::abs(s.objective() - oracle) < 1e-6 ? "(ok)" : "(FAIL)");
    if (std::abs(s.objective() - oracle) >= 1e-6) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
