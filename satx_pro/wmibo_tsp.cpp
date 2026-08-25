#include <satx/satx.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>

using namespace satx::solver::wmibo;

// TSP (viajante de comercio) resuelto con WMIBO (modelo híbrido
// booleano-lineal): formulación MTZ con 20 variables booleanas (aristas
// dirigidas x_ij) y 5 variables reales (posiciones u_i para eliminar
// subtours). El kernel combina branch-and-bound sobre las booleanas con
// relajación LP.
//
//   min  Σ d_ij·x_ij
//   s.a. Σ_j x_ij = 1            ∀i       (una salida por ciudad)
//        Σ_i x_ij = 1            ∀j       (una entrada por ciudad)
//        u_i − u_j + n·x_ij ≤ n−1  ∀i,j≠0 (MTZ: sin subtours)
//        u_0 = 0
//        x_ij binarias, u_i ∈ [0, n]
//
// El resultado se verifica contra la búsqueda exhaustiva del host (5! = 120
// permutaciones).

int main() {
    constexpr int n = 5;
    // Coordenadas enteras (x, y) de las 5 ciudades.
    const double cx[n] = {0.0, 3.0, 4.0, 1.0, 0.0};
    const double cy[n] = {0.0, 0.0, 3.0, 4.0, 2.0};
    int d[n][n] = {};
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) d[i][j] = static_cast<int>(std::lround(std::hypot(cx[i] - cx[j], cy[i] - cy[j])));

    model m{"tsp5"};

    // x_ij: variable booleana por arista dirigida i→j (i != j).
    std::vector<variable*> x;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) x.push_back(&m.add_boolean("x" + std::to_string(i) + "_" + std::to_string(j)));
    const auto xv = [&](int i, int j) -> variable& {
        int k = 0;
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
                if (a != b) {
                    if (a == i && b == j) return *x[k];
                    ++k;
                }
        std::abort();
    };

    // u_i: posición en el tour (MTZ), u_0 = 0.
    std::vector<variable*> u;
    for (int i = 0; i < n; ++i) u.push_back(&m.add_real("u" + std::to_string(i), 0.0, n));

    // Grado: una salida y una entrada por ciudad.
    for (int i = 0; i < n; ++i) {
        expr out, in;
        for (int j = 0; j < n; ++j) {
            if (i != j) out += xv(i, j);
            if (j != i) in += xv(j, i);
        }
        m.add_constraint(out, compare::eq, 1.0);
        m.add_constraint(in, compare::eq, 1.0);
    }

    // MTZ: u_i − u_j + n·x_ij <= n−1  (para i, j ≠ 0).
    for (int i = 1; i < n; ++i)
        for (int j = 1; j < n; ++j)
            if (i != j) m.add_constraint((*u[i]) - (*u[j]) + static_cast<double>(n) * xv(i, j),
                                         compare::le, static_cast<double>(n - 1));

    // u_0 = 0.
    m.add_constraint(*u[0], compare::eq, 0.0);

    // Objetivo: minimizar la distancia total.
    expr cost;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) cost += static_cast<double>(d[i][j]) * xv(i, j);
    m.set_objective(cost, sense::min);

    const auto s = m.solve();

    std::printf("estado: %s\n",
                s.state() == status::optimal ? "OPTIMAL"
                : s.state() == status::infeasible ? "INFEASIBLE" : "UNKNOWN");
    if (!s.has_values()) return EXIT_FAILURE;

    std::printf("distancia óptima (wmibo) = %.0f\n", s.objective());

    // Reconstrucción del tour desde la ciudad 0 siguiendo x_ij = 1.
    std::vector<int> tour = {0};
    int cur = 0;
    for (int step = 0; step < n - 1; ++step) {
        for (int j = 0; j < n; ++j) {
            if (j == cur) continue;
            if (s.boolean(xv(cur, j))) {
                tour.push_back(j);
                cur = j;
                break;
            }
        }
    }
    std::printf("tour: ");
    for (int c : tour) std::printf("%d ", c);
    std::printf("(vuelta a %d)\n", tour.front());

    // Verificación contra el oráculo: búsqueda exhaustiva sobre las 5!
    // permutaciones.
    int oracle = std::numeric_limits<int>::max();
    std::vector<int> perm = {0, 1, 2, 3, 4};
    do {
        int cost_perm = 0;
        for (int k = 0; k < n; ++k) cost_perm += d[perm[k]][perm[(k + 1) % n]];
        oracle = std::min(oracle, cost_perm);
    } while (std::next_permutation(perm.begin(), perm.end()));

    std::printf("distancia óptima (oráculo) = %d %s\n", oracle,
                std::abs(s.objective() - static_cast<double>(oracle)) < 1e-6 ? "(ok)" : "(FAIL)");
    if (std::abs(s.objective() - static_cast<double>(oracle)) >= 1e-6) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
