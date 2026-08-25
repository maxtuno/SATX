// cross_docking — catálogo 1.5: cross-docking con transbordo, kernel SLIME.
//
// Camiones de entrada con cargas de productos y camiones de salida con
// pedidos. Variables z[i][j][p]: la carga p del camión de entrada i alimenta
// la salida j. Restricciones:
//   · balance: cada salida recibe exactamente lo que pide; cada entrada
//     despacha como máximo lo que trae;
//   · precedencia: si z[i][j][p] entonces salida_j parte después de la
//     llegada de entrada_i (s_j ≥ a_i);
//   · puerta única de salidas: tiempos de salida todos distintos.
//
// Objetivo (búsqueda binaria sobre el makespan con SLIME): minimizar el
// instante máximo de salida. Se verifica contra el oráculo del host.

#include <satx/satx.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int IN = 3, OUT = 3, P = 3;  // entradas, salidas, productos
    const int arr[IN] = {0, 4, 8};         // llegadas de las entradas
    // entradas: carga por producto (0/1)
    const int load[IN][P] = {
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0},
    };
    // salidas: demanda por producto (0/1)
    const int need[OUT][P] = {
        {1, 1, 0},
        {1, 0, 1},
        {0, 1, 1},
    };

    using I = satx::integer<6>;  // tiempo 0..24

    const auto build = [&](satx::engine& e, int B, std::vector<I>& dep) {
        std::vector<std::vector<std::vector<satx::lit_t>>> z(
            IN, std::vector<std::vector<satx::lit_t>>(OUT, std::vector<satx::lit_t>(P)));
        for (int i = 0; i < IN; ++i)
            for (int j = 0; j < OUT; ++j)
                for (int p = 0; p < P; ++p) z[i][j][p] = e.add_variable();

        for (int j = 0; j < OUT; ++j) {
            dep.emplace_back(e);
            e.add_unit(satx::le_lit(e, I{0}, dep[j]));
            e.add_unit(satx::le_lit(e, dep[j], I{B}));
        }

        // balance de salidas: Σ_i z[i][j][p] == need[j][p]
        for (int j = 0; j < OUT; ++j)
            for (int p = 0; p < P; ++p) {
                if (need[j][p] == 0) {
                    for (int i = 0; i < IN; ++i) e.add_unit(-z[i][j][p]);
                } else {
                    e.add_clause({z[0][j][p], z[1][j][p], z[2][j][p]});
                    for (int i1 = 0; i1 < IN; ++i1)
                        for (int i2 = i1 + 1; i2 < IN; ++i2)
                            e.add_unit(-satx::gates::and2(e, z[i1][j][p], z[i2][j][p]));
                }
            }
        // balance de entradas: Σ_j z[i][j][p] ≤ load[i][p]
        for (int i = 0; i < IN; ++i)
            for (int p = 0; p < P; ++p) {
                if (load[i][p] == 0)
                    for (int j = 0; j < OUT; ++j) e.add_unit(-z[i][j][p]);
                else
                    for (int j1 = 0; j1 < OUT; ++j1)
                        for (int j2 = j1 + 1; j2 < OUT; ++j2)
                            e.add_unit(-satx::gates::and2(e, z[i][j1][p], z[i][j2][p]));
            }
        // precedencia: z[i][j][p] → dep[j] ≥ arr[i]
        for (int i = 0; i < IN; ++i)
            for (int j = 0; j < OUT; ++j)
                for (int p = 0; p < P; ++p)
                    e.add_clause({-z[i][j][p], satx::ge_lit(e, dep[j], I{arr[i]})});
        // puerta única: salidas a instantes distintos
        for (int j1 = 0; j1 < OUT; ++j1)
            for (int j2 = j1 + 1; j2 < OUT; ++j2)
                e.add_unit(satx::ne_lit(e, dep[j1], dep[j2]));
    };

    // búsqueda binaria del makespan mínimo
    int lo = 0, hi = 24;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        satx::engine e;
        std::vector<I> dep;
        build(e, mid, dep);
        if (satx::solver::solve(e)) hi = mid;
        else lo = mid + 1;
    }
    std::printf("makespan mínimo (SLIME) = %d\n", lo);

    // ── oráculo del host ────────────────────────────────────────────────────
    // Cada demanda (j, p) elige una entrada con la mercancía; cada (i, p)
    // puede alimentar a lo sumo una salida; tiempos de salida = enteros
    // distintos ≥ cotas (fuerza bruta en [0, 24]), minimizando el máximo.
    int oracle = 1000;
    std::vector<std::vector<int>> choices(OUT * P);
    for (int j = 0; j < OUT; ++j)
        for (int p = 0; p < P; ++p) {
            if (need[j][p] == 0) continue;
            for (int i = 0; i < IN; ++i)
                if (load[i][p] == 1) choices[j * P + p].push_back(i);
        }
    std::vector<int> idx(OUT * P, 0);
    while (true) {
        // cota inferior de salida por camión y capacidad de entrada
        std::vector<int> lb(OUT, 0);
        std::vector<int> used(IN * P, 0);
        bool valid = true;
        for (int j = 0; j < OUT && valid; ++j)
            for (int p = 0; p < P && valid; ++p) {
                if (need[j][p] == 0) continue;
                const int i = choices[j * P + p][idx[j * P + p]];
                if (++used[i * P + p] > load[i][p]) { valid = false; break; }
                lb[j] = std::max(lb[j], arr[i]);
            }
        if (valid) {
            // tiempos de salida distintos en [0, 24]: fuerza bruta
            for (int t0 = lb[0]; t0 <= 24; ++t0)
                for (int t1 = lb[1]; t1 <= 24; ++t1) {
                    if (t1 == t0) continue;
                    for (int t2 = lb[2]; t2 <= 24; ++t2) {
                        if (t2 == t0 || t2 == t1) continue;
                        oracle = std::min(oracle, std::max({t0, t1, t2}));
                    }
                }
        }
        // siguiente combinación (odómetro sobre las demandas no vacías)
        int k = OUT * P - 1;
        while (k >= 0 && (choices[k].empty() || idx[k] + 1 >= static_cast<int>(choices[k].size()))) {
            if (choices[k].empty()) { --k; continue; }
            idx[k] = 0;
            --k;
        }
        if (k < 0) break;
        ++idx[k];
    }

    std::printf("makespan mínimo (oráculo) = %d %s\n", oracle, lo == oracle ? "(ok)" : "(FAIL)");
    if (lo != oracle) return EXIT_FAILURE;

    // instancia con makespan óptimo: mostrar el plan
    satx::engine e;
    std::vector<I> dep;
    build(e, lo, dep);
    const auto sol = satx::solver::solve(e);
    if (!sol) return EXIT_FAILURE;
    std::printf("plan: salidas en t =");
    for (int j = 0; j < OUT; ++j) std::printf(" %d", static_cast<int>(dep[j].value(*sol)));
    std::printf("\n");
    return EXIT_SUCCESS;
}
