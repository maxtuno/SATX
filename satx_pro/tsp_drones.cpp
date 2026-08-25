// tsp_drones — catálogo 2.5: reparto con drones (TSP-D), kernel WMIBO.
//
// Un camión con un dron entrega paquetes. El dron despega del depósito,
// entrega un paquete y vuelve (simplificación de lanzamiento/recogida en
// paradas); el camión recorre los paquetes restantes. El makespan es el
// máximo entre el tiempo del camión y el tiempo total del dron:
//
//   min  z
//   s.a. z ≥ Σ t_ij·x_ij                  (tiempo del camión)
//        z ≥ Σ_p 2·f_p·d_p                (tiempo total del dron)
//        visited_p ↔ ¬d_p, MTZ sobre visitados,
//        indicadores de grado reificados por visited_p
//
// Verificación: oráculo del host — 2^4 subconjuntos × permutaciones.

#include <satx/satx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace satx::solver::wmibo;

int main() {
    constexpr int n = 4;  // paquetes
    const double cx[n] = {3.0, 5.0, 2.0, 6.0};
    const double cy[n] = {2.0, 1.0, 5.0, 4.0};

    int t[n + 1][n + 1] = {};
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= n; ++j)
            if (i != j) {
                const double x1 = i == 0 ? 0.0 : cx[i - 1];
                const double y1 = i == 0 ? 0.0 : cy[i - 1];
                const double x2 = j == 0 ? 0.0 : cx[j - 1];
                const double y2 = j == 0 ? 0.0 : cy[j - 1];
                t[i][j] = static_cast<int>(std::lround(std::hypot(x1 - x2, y1 - y2)));
            }
    int f[n] = {};
    for (int p = 0; p < n; ++p)
        f[p] = static_cast<int>(std::lround(std::hypot(cx[p], cy[p])));

    model m{"tsp_drones4"};

    // visited_p: el camión visita p; d_p: el dron entrega p.
    std::vector<variable*> visited, d;
    for (int p = 0; p < n; ++p) {
        visited.push_back(&m.add_boolean("v" + std::to_string(p)));
        d.push_back(&m.add_boolean("d" + std::to_string(p)));
        // visited ↔ ¬d
        m.add_hard_clause({lit{*visited[p]}, lit{*d[p]}});      // visited ∨ d
        m.add_hard_clause({~lit{*visited[p]}, ~lit{*d[p]}});    // ¬visited ∨ ¬d
    }

    // arcos del camión
    std::vector<std::vector<variable*>> x(n + 1, std::vector<variable*>(n + 1));
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= n; ++j)
            if (i != j) x[i][j] = &m.add_boolean("x" + std::to_string(i) + "_" + std::to_string(j));

    // u_i: orden en la ruta (MTZ)
    std::vector<variable*> u;
    for (int i = 1; i <= n; ++i) u.push_back(&m.add_real("u" + std::to_string(i), 0.0, n));

    // depósito: entra y sale siempre
    {
        expr out, in;
        for (int j = 1; j <= n; ++j) {
            out += *x[0][j];
            in += *x[j][0];
        }
        m.add_constraint(out, compare::eq, 1.0);
        m.add_constraint(in, compare::eq, 1.0);
    }
    // grado reificado: visited_p ⇒ (Σ x = 1), ¬visited_p ⇒ (Σ x = 0)
    for (int p = 1; p <= n; ++p) {
        expr out, in;
        for (int j = 0; j <= n; ++j)
            if (p != j) {
                out += *x[p][j];
                in += *x[j][p];
            }
        const variable& v = *visited[p - 1];
        m.add_indicator(lit{v}, out, compare::ge, 1.0);
        m.add_indicator(lit{v}, out, compare::le, 1.0);
        m.add_indicator(lit{v}, in, compare::ge, 1.0);
        m.add_indicator(lit{v}, in, compare::le, 1.0);
        m.add_indicator(~lit{v}, out, compare::le, 0.0);
        m.add_indicator(~lit{v}, in, compare::le, 0.0);
    }
    // MTZ
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
            if (i != j)
                m.add_constraint((*u[i - 1]) - (*u[j - 1]) + static_cast<double>(n) * (*x[i][j]),
                                 compare::le, static_cast<double>(n - 1));

    // tiempos
    expr truck_t, drone_t;
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= n; ++j)
            if (i != j) truck_t += static_cast<double>(t[i][j]) * (*x[i][j]);
    for (int p = 0; p < n; ++p) drone_t += static_cast<double>(2 * f[p]) * (*d[p]);

    variable& z = m.add_real("z", 0.0, 100.0);
    m.add_constraint(z - truck_t, compare::ge, 0.0);
    m.add_constraint(z - drone_t, compare::ge, 0.0);
    m.set_objective(z, sense::min);

    const auto s = m.solve();

    std::printf("estado: %s\n",
                s.state() == status::optimal ? "OPTIMAL"
                : s.state() == status::infeasible ? "INFEASIBLE" : "UNKNOWN");
    if (s.state() != status::optimal || !s.has_values()) return EXIT_FAILURE;

    std::printf("makespan óptimo (wmibo) = %.0f\n", s.objective());
    std::printf("  dron entrega:");
    for (int p = 0; p < n; ++p)
        if (s.boolean(*d[p])) std::printf(" p%d", p);
    std::printf("\n  camión:");
    int cur = 0;
    while (true) {
        int nxt = -1;
        for (int j = 0; j <= n; ++j)
            if (j != cur && x[cur][j] != nullptr && s.boolean(*x[cur][j])) { nxt = j; break; }
        if (nxt == -1 || nxt == 0) break;
        std::printf(" → %d", nxt);
        cur = nxt;
    }
    std::printf("\n");

    // ── oráculo del host ────────────────────────────────────────────────────
    double oracle = std::numeric_limits<double>::infinity();
    for (int mask = 0; mask < (1 << n); ++mask) {
        std::vector<int> stops;  // paquetes del camión
        double drone_tm = 0.0;
        for (int p = 0; p < n; ++p) {
            if (mask & (1 << p)) stops.push_back(p + 1);
            else drone_tm += 2.0 * f[p];
        }
        std::vector<int> order = stops;
        do {
            double truck_tm = t[0][order.empty() ? 0 : order[0]];
            for (std::size_t i = 1; i < order.size(); ++i) truck_tm += t[order[i - 1]][order[i]];
            if (!order.empty()) truck_tm += t[order.back()][0];
            oracle = std::min(oracle, std::max(truck_tm, drone_tm));
        } while (std::next_permutation(order.begin(), order.end()));
    }

    std::printf("makespan óptimo (oráculo) = %.0f %s\n", oracle,
                std::abs(s.objective() - oracle) < 1e-6 ? "(ok)" : "(FAIL)");
    if (std::abs(s.objective() - oracle) >= 1e-6) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
