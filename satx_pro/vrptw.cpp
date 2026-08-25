// vrptw — catálogo 2.1: reparto con ventanas de tiempo y flota heterogénea,
// kernel WMIBO. Formulación MTZ con continuidad temporal big-M (la del
// catálogo):
//
//   min  Σ t_ij·x_ijk
//   s.a. τ_j ≥ τ_i + p_i + t_ij − M(1 − x_ijk)   (continuidad, big-M)
//        e_i ≤ τ_i ≤ l_i                          (ventana)
//        Σ_{k,j} x_ijk = 1                        (cobertura)
//        Σ_{i≠0} d_i·Σ_j x_ijk ≤ Q_k              (capacidad)
//        u_i − u_j + n·x_ijk ≤ n−1                (MTZ)
//
// Verificación: oráculo del host — permutaciones partidas en rutas con
// comprobación de ventanas (llegada lo antes posible).

#include <satx/satx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace satx::solver::wmibo;

int main() {
    constexpr int n = 5;         // clientes
    constexpr int K = 2;         // vehículos
    constexpr int Q = 12;
    const double d[n] = {3.0, 4.0, 2.0, 5.0, 4.0};
    const double e[n] = {0.0, 2.0, 4.0, 6.0, 8.0};   // inicio de ventana
    const double l[n] = {10.0, 12.0, 14.0, 16.0, 18.0};
    const double psv[n] = {1.0, 1.0, 1.0, 1.0, 1.0};  // servicio
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

    model m{"vrptw5"};

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

    std::vector<variable*> u, tau;
    for (int i = 1; i <= n; ++i) {
        u.push_back(&m.add_real("u" + std::to_string(i), 0.0, n));
        tau.push_back(&m.add_real("t" + std::to_string(i), 0.0, 20.0));
        m.add_constraint(*tau[i - 1], compare::ge, e[i - 1]);
        m.add_constraint(*tau[i - 1], compare::le, l[i - 1]);
    }

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
    for (int k = 0; k < K; ++k) {
        expr cap;
        for (int i = 1; i <= n; ++i)
            for (int j = 0; j <= n; ++j)
                if (i != j) cap += d[i - 1] * (*x[i][j][k]);
        m.add_constraint(cap, compare::le, Q);
    }
    for (int k = 0; k < K; ++k)
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= n; ++j)
                if (i != j)
                    m.add_constraint((*u[i - 1]) - (*u[j - 1]) + static_cast<double>(n) * (*x[i][j][k]),
                                     compare::le, static_cast<double>(n - 1));
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

    // continuidad temporal con big-M (formulación del catálogo):
    //   τ_j ≥ τ_i + p_i + t_ij − M·(1 − x_ijk)   (τ_0 = 0)
    // M debe cumplir M ≥ max τ + max(p+t) = 20 + 13 = 33 para que la
    // restricción sea vacua cuando x_ijk = 0.
    constexpr double M = 40.0;
    for (int k = 0; k < K; ++k)
        for (int i = 0; i <= n; ++i)
            for (int j = 1; j <= n; ++j)
                if (i != j) {
                    const double rhs = (i == 0 ? 0.0 : psv[i - 1]) + t[i][j];
                    expr lhs = *tau[j - 1];
                    if (i != 0) lhs -= *tau[i - 1];
                    m.add_constraint(lhs - M * (*x[i][j][k]), compare::ge, rhs - M);
                }

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

    std::printf("costo óptimo (wmibo) = %.0f\n", s.objective());

    // ── oráculo del host ────────────────────────────────────────────────────
    // permutaciones × cortes con ventanas y capacidad
    double oracle = std::numeric_limits<double>::infinity();
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i + 1;
    do {
        for (int cut = 0; cut <= n; ++cut) {
            double loadA = 0.0, loadB = 0.0;
            for (int i = 0; i < cut; ++i) loadA += d[perm[i] - 1];
            for (int i = cut; i < n; ++i) loadB += d[perm[i] - 1];
            if (loadA > Q || loadB > Q) continue;
            // ruta A: llegadas con espera
            double prev = 0.0, tauA = 0.0;
            double cA = 0.0;
            bool okA = true;
            for (int i = 0; i < cut && okA; ++i) {
                tauA = std::max(tauA + t[static_cast<int>(prev)][perm[i]], e[perm[i] - 1]);
                if (tauA > l[perm[i] - 1]) okA = false;
                cA += t[static_cast<int>(prev)][perm[i]];
                prev = perm[i];
            }
            if (okA) cA += t[static_cast<int>(prev)][0];
            // ruta B
            prev = 0.0;
            double tauB = 0.0, cB = 0.0;
            bool okB = true;
            for (int i = cut; i < n && okB; ++i) {
                tauB = std::max(tauB + t[static_cast<int>(prev)][perm[i]], e[perm[i] - 1]);
                if (tauB > l[perm[i] - 1]) okB = false;
                cB += t[static_cast<int>(prev)][perm[i]];
                prev = perm[i];
            }
            if (okB) cB += t[static_cast<int>(prev)][0];
            if (okA && okB) oracle = std::min(oracle, cA + cB);
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    std::printf("costo óptimo (oráculo) = %.0f %s\n", oracle,
                std::abs(s.objective() - oracle) < 1e-6 ? "(ok)" : "(FAIL)");
    if (std::abs(s.objective() - oracle) >= 1e-6) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
