// co2_routes — catálogo 5.11: rutas con métricas de sostenibilidad (CO₂),
// kernel WMIBO.
//
// Flota heterogénea (eléctrico/diésel) con capacidad; objetivo multi-criterio
// distancia + CO₂ (emisión por km y vehículo) + penalización blanda por
// atender clientes de zona de bajas emisiones (LEZ) con el vehículo diésel.
//
//   min  w1·Σ t_ij·x_ijk + w2·Σ e_k·t_ij·x_ijk + Σ soft
//
// Verificación: recomputación de rutas, capacidad y objetivo en el host, y
// oráculo por permutaciones × cortes × asignación de vehículos.

#include <satx/satx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace satx::solver::wmibo;

int main() {
    constexpr int n = 4;   // clientes
    constexpr int K = 2;   // vehículos
    constexpr int Q = 8;   // capacidad
    const double d[n] = {2.0, 3.0, 2.0, 3.0};
    const bool lez[n] = {1, 1, 0, 0};  // clientes en zona de bajas emisiones
    const double emiss[K] = {0.5, 1.5};  // kg CO₂ por unidad de distancia
    constexpr double W1 = 1.0, W2 = 0.01, PENALTY = 5.0;
    const double cx[n] = {3.0, 6.0, 1.0, 8.0};
    const double cy[n] = {0.0, 1.0, 5.0, 3.0};

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

    model m{"co2_routes"};

    std::vector<std::vector<std::vector<variable*>>> x(
        n + 1, std::vector<std::vector<variable*>>(n + 1, std::vector<variable*>(K)));
    for (int k = 0; k < K; ++k)
        for (int i = 0; i <= n; ++i)
            for (int j = 0; j <= n; ++j)
                if (i != j)
                    x[i][j][k] = &m.add_boolean("x" + std::to_string(i) + "_" + std::to_string(j) +
                                                "_" + std::to_string(k));
    std::vector<variable*> y;
    for (int k = 0; k < K; ++k) y.push_back(&m.add_boolean("y" + std::to_string(k)));
    std::vector<variable*> u;
    for (int i = 1; i <= n; ++i) u.push_back(&m.add_real("u" + std::to_string(i), 0.0, n));

    // cobertura
    for (int i = 1; i <= n; ++i) {
        expr out, in;
        for (int k = 0; k < K; ++k)
            for (int j = 0; j <= n; ++j)
                if (i != j) {
                    out += *x[i][j][k];
                    in += *x[j][i][k];
                }
        m.add_constraint(out, compare::eq, 1.0);
        m.add_constraint(in, compare::eq, 1.0);
    }
    // depósito
    for (int k = 0; k < K; ++k) {
        expr out, in;
        for (int j = 1; j <= n; ++j) {
            out += *x[0][j][k];
            in += *x[j][0][k];
        }
        m.add_constraint(out - (*y[k]), compare::eq, 0.0);
        m.add_constraint(in - (*y[k]), compare::eq, 0.0);
    }
    // acoplamiento: arcos solo en vehículos usados
    for (int k = 0; k < K; ++k)
        for (int i = 0; i <= n; ++i)
            for (int j = 0; j <= n; ++j)
                if (i != j)
                    m.add_constraint((*x[i][j][k]) - (*y[k]), compare::le, 0.0);
    // conservación de flujo por vehículo (rutas coherentes por vehículo)
    for (int k = 0; k < K; ++k)
        for (int i = 1; i <= n; ++i) {
            expr out, in;
            for (int j = 0; j <= n; ++j)
                if (i != j) {
                    out += *x[i][j][k];
                    in += *x[j][i][k];
                }
            m.add_constraint(out - in, compare::eq, 0.0);
        }
    // capacidad
    for (int k = 0; k < K; ++k) {
        expr cap;
        for (int i = 1; i <= n; ++i)
            for (int j = 0; j <= n; ++j)
                if (i != j) cap += d[i - 1] * (*x[i][j][k]);
        m.add_constraint(cap, compare::le, Q);
    }
    // MTZ
    for (int k = 0; k < K; ++k)
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= n; ++j)
                if (i != j)
                    m.add_constraint((*u[i - 1]) - (*u[j - 1]) + static_cast<double>(n) * (*x[i][j][k]),
                                     compare::le, static_cast<double>(n - 1));
    // objetivo: distancia + CO₂
    expr obj;
    for (int k = 0; k < K; ++k)
        for (int i = 0; i <= n; ++i)
            for (int j = 0; j <= n; ++j)
                if (i != j) obj += (W1 + W2 * emiss[k]) * static_cast<double>(t[i][j]) * (*x[i][j][k]);
    m.set_objective(obj, sense::min);
    // blandas: cliente LEZ atendido por el vehículo eléctrico (0)
    for (int j = 1; j <= n; ++j)
        if (lez[j - 1]) {
            std::vector<lit> lits;
            for (int i = 0; i <= n; ++i)
                if (i != j) lits.emplace_back(*x[i][j][0]);
            m.add_soft_clause(PENALTY, lits);
        }

    const auto s = m.solve();

    std::printf("estado: %s\n",
                s.state() == status::optimal ? "OPTIMAL"
                : s.state() == status::infeasible ? "INFEASIBLE" : "UNKNOWN");
    if (s.state() != status::optimal || !s.has_values()) return EXIT_FAILURE;

    std::printf("objetivo óptimo (wmibo) = %.3f\n", s.objective());

    // ── verificación en el host ─────────────────────────────────────────────
    double verified = 0.0;
    int penalties = 0;
    std::vector<int> visited(n, 0);
    for (int k = 0; k < K; ++k) {
        if (!s.boolean(*y[k])) continue;
        std::vector<int> route = {0};
        int cur = 0;
        while (true) {
            int nxt = -1;
            for (int j = 0; j <= n; ++j)
                if (j != cur && x[cur][j][k] != nullptr && s.boolean(*x[cur][j][k])) { nxt = j; break; }
            if (nxt == -1) break;
            route.push_back(nxt);
            cur = nxt;
            if (cur == 0) break;  // vuelta al depósito: la ruta termina
        }
        double load = 0.0;
        std::printf("  vehículo %d (%s): 0", k, k == 0 ? "eléctrico" : "diésel");
        for (std::size_t i = 1; i < route.size(); ++i) {
            std::printf(" → %d", route[i]);
            if (route[i] != 0) {
                load += d[route[i] - 1];
                ++visited[route[i] - 1];
                if (k == 1 && lez[route[i] - 1]) ++penalties;
            }
            verified += (W1 + W2 * emiss[k]) * t[route[i - 1]][route[i]];
        }
        std::printf(" (carga %g/%d)\n", load, Q);
        if (load > Q + 1e-9) {
            std::printf("VERIFICACIÓN FALLIDA: capacidad\n");
            return EXIT_FAILURE;
        }
    }
    for (int i = 0; i < n; ++i)
        if (visited[i] != 1) {
            std::printf("VERIFICACIÓN FALLIDA: cliente %d visitado %d veces\n", i + 1, visited[i]);
            return EXIT_FAILURE;
        }
    verified += static_cast<double>(penalties) * PENALTY;
    std::printf("objetivo verificado = %.3f (%d penalizaciones) %s\n", verified, penalties,
                std::abs(verified - s.objective()) < 1e-6 ? "(ok)" : "(FAIL)");
    if (std::abs(verified - s.objective()) >= 1e-6) return EXIT_FAILURE;

    // ── oráculo del host ────────────────────────────────────────────────────
    double oracle = std::numeric_limits<double>::infinity();
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i + 1;
    do {
        for (int cut = 0; cut <= n; ++cut) {
            double loadA = 0.0, loadB = 0.0;
            for (int i = 0; i < cut; ++i) loadA += d[perm[i] - 1];
            for (int i = cut; i < n; ++i) loadB += d[perm[i] - 1];
            if (loadA > Q || loadB > Q) continue;
            // ruta A en vehículo 0 o 1; ruta B en el otro
            const int va[2][2] = {{0, 1}, {1, 0}};
            for (int swap = 0; swap < 2; ++swap) {
                double c = 0.0;
                int pen = 0;
                int prev = 0;
                for (int i = 0; i < cut; ++i) {
                    c += (W1 + W2 * emiss[va[swap][0]]) * t[prev][perm[i]];
                    if (va[swap][0] == 1 && lez[perm[i] - 1]) ++pen;
                    prev = perm[i];
                }
                c += (W1 + W2 * emiss[va[swap][0]]) * t[prev][0];
                prev = 0;
                for (int i = cut; i < n; ++i) {
                    c += (W1 + W2 * emiss[va[swap][1]]) * t[prev][perm[i]];
                    if (va[swap][1] == 1 && lez[perm[i] - 1]) ++pen;
                    prev = perm[i];
                }
                c += (W1 + W2 * emiss[va[swap][1]]) * t[prev][0];
                oracle = std::min(oracle, c + pen * PENALTY);
            }
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    std::printf("objetivo óptimo (oráculo) = %.3f %s\n", oracle,
                std::abs(s.objective() - oracle) < 1e-6 ? "(ok)" : "(FAIL)");
    if (std::abs(s.objective() - oracle) >= 1e-6) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
