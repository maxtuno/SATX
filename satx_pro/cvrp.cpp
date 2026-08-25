// cvrp — catálogo 1.2: rutas de vehículos con capacidad (CVRP), kernel WMIBO
// (modelo híbrido booleano-lineal con MTZ para eliminar subciclos).
//
//   min  Σ_{k,i,j} t_{i,j}·x_{i,j,k}
//   s.a. Σ_{k,j} x_{i,j,k} = 1            ∀i cliente   (atendido una vez)
//        Σ_j x_{0,j,k} = Σ_i x_{i,0,k} = y_k            (uso de vehículo k)
//        Σ_{i≠0} d_i·Σ_j x_{i,j,k} ≤ Q     ∀k           (capacidad)
//        u_i − u_j + n·x_{i,j,k} ≤ n−1     ∀i,j≠0, k    (MTZ)
//
// Verificación: oráculo del host — permutaciones de clientes partidas en dos
// rutas (5! × 6 cortes), filtrando capacidad.

#include <satx/satx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace satx::solver::wmibo;

int main() {
    constexpr int n = 5;        // clientes
    constexpr int K = 2;        // vehículos
    constexpr int Q = 10;       // capacidad
    const double d[n] = {3.0, 4.0, 2.0, 5.0, 4.0};
    const double cx[n] = {3.0, 6.0, 1.0, 8.0, 4.0};
    const double cy[n] = {0.0, 1.0, 5.0, 3.0, 7.0};

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

    model m{"cvrp5"};

    // x_{i,j,k}: el vehículo k viaja i→j (i != j).
    std::vector<std::vector<std::vector<variable*>>> x(
        n + 1, std::vector<std::vector<variable*>>(n + 1, std::vector<variable*>(K)));
    for (int k = 0; k < K; ++k)
        for (int i = 0; i <= n; ++i)
            for (int j = 0; j <= n; ++j)
                if (i != j)
                    x[i][j][k] = &m.add_boolean("x" + std::to_string(i) + "_" + std::to_string(j) +
                                                "_" + std::to_string(k));

    // y_k: vehículo usado.
    std::vector<variable*> y;
    for (int k = 0; k < K; ++k) y.push_back(&m.add_boolean("y" + std::to_string(k)));

    // u_i: posición en la ruta (MTZ), solo clientes.
    std::vector<variable*> u;
    for (int i = 1; i <= n; ++i) u.push_back(&m.add_real("u" + std::to_string(i), 0.0, n));

    // Cada cliente atendido exactamente una vez.
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

    // Depósito: el vehículo k sale y vuelve si y solo si se usa.
    for (int k = 0; k < K; ++k) {
        expr out, in;
        for (int j = 1; j <= n; ++j) {
            out += *x[0][j][k];
            in += *x[j][0][k];
        }
        m.add_constraint(out - (*y[k]), compare::eq, 0.0);
        m.add_constraint(in - (*y[k]), compare::eq, 0.0);
    }

    // Acoplamiento: arcos de un vehículo solo si el vehículo se usa.
    for (int k = 0; k < K; ++k)
        for (int i = 0; i <= n; ++i)
            for (int j = 0; j <= n; ++j)
                if (i != j)
                    m.add_constraint((*x[i][j][k]) - (*y[k]), compare::le, 0.0);

    // Conservación de flujo por vehículo: en cada cliente, las entradas del
    // vehículo k igualan sus salidas (sin esto las rutas se entrecruzan:
    // un vehículo puede «entrar» a un cliente y otro «salirlo»).
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

    // Capacidad por vehículo.
    for (int k = 0; k < K; ++k) {
        expr cap;
        for (int i = 1; i <= n; ++i)
            for (int j = 0; j <= n; ++j)
                if (i != j) cap += d[i - 1] * (*x[i][j][k]);
        m.add_constraint(cap, compare::le, Q);
    }

    // MTZ: u_i − u_j + n·x_{i,j,k} ≤ n−1 (clientes i,j).
    for (int k = 0; k < K; ++k)
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= n; ++j)
                if (i != j)
                    m.add_constraint((*u[i - 1]) - (*u[j - 1]) + static_cast<double>(n) * (*x[i][j][k]),
                                     compare::le, static_cast<double>(n - 1));

    // Objetivo: distancia total.
    expr cost;
    for (int k = 0; k < K; ++k)
        for (int i = 0; i <= n; ++i)
            for (int j = 0; j <= n; ++j)
                if (i != j) cost += static_cast<double>(t[i][j]) * (*x[i][j][k]);
    m.set_objective(cost, sense::min);

    const auto s = m.solve();

    std::printf("estado: %s\n",
                s.state() == status::optimal ? "OPTIMAL"
                : s.state() == status::infeasible ? "INFEASIBLE" : "UNKNOWN");
    if (s.state() != status::optimal || !s.has_values()) return EXIT_FAILURE;

    std::printf("distancia óptima (wmibo) = %.0f\n", s.objective());

    // Reconstrucción de las rutas.
    double verified_cost = 0.0;
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
        std::printf("  vehículo %d: 0", k);
        for (std::size_t i = 1; i < route.size(); ++i) {
            std::printf(" → %d", route[i]);
            if (route[i] != 0) { load += d[route[i] - 1]; ++visited[route[i] - 1]; }
            verified_cost += t[route[static_cast<std::size_t>(i - 1)]][route[i]];
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
    std::printf("costo verificado = %.0f %s\n", verified_cost,
                std::abs(verified_cost - s.objective()) < 1e-6 ? "(ok)" : "(FAIL)");
    if (std::abs(verified_cost - s.objective()) >= 1e-6) return EXIT_FAILURE;

    // ── oráculo del host ────────────────────────────────────────────────────
    // Cada permutación de clientes partida en (ruta A, ruta B) con un corte.
    int oracle = std::numeric_limits<int>::max();
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i + 1;
    do {
        for (int cut = 0; cut <= n; ++cut) {
            double loadA = 0.0, loadB = 0.0;
            for (int i = 0; i < cut; ++i) loadA += d[perm[i] - 1];
            for (int i = cut; i < n; ++i) loadB += d[perm[i] - 1];
            if (loadA > Q || loadB > Q) continue;
            int c = 0, prev = 0;
            for (int i = 0; i < cut; ++i) { c += t[prev][perm[i]]; prev = perm[i]; }
            c += t[prev][0];
            prev = 0;
            for (int i = cut; i < n; ++i) { c += t[prev][perm[i]]; prev = perm[i]; }
            c += t[prev][0];
            oracle = std::min(oracle, c);
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    std::printf("distancia óptima (oráculo) = %d %s\n", oracle,
                std::abs(s.objective() - static_cast<double>(oracle)) < 1e-6 ? "(ok)" : "(FAIL)");
    if (std::abs(s.objective() - static_cast<double>(oracle)) >= 1e-6) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
