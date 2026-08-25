// robot_path — catálogo 5.9: planificación de movimientos de robot
// (discretizada), kernel SLIME.
//
// Rejilla 4×4 con obstáculos; el robot camina del inicio a la meta en T
// pasos. Se busca el T mínimo con búsqueda binaria (reconstruyendo la
// fórmula) y se compara contra el BFS del host.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <queue>
#include <vector>

int main() {
    constexpr int G = 4;
    const int sx = 0, sy = 0, gx = 3, gy = 3;
    // obstáculos
    const bool wall[G][G] = {
        {0, 1, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 1},
        {0, 1, 0, 0},
    };
    const auto cell = [&](int x, int y) { return y * G + x; };

    const auto build = [&](satx::engine& e, int T,
                           std::vector<std::vector<satx::lit_t>>& p) {
        for (int t = 0; t <= T; ++t)
            for (int c = 0; c < G * G; ++c) p[t][c] = e.add_variable();
        e.add_unit(p[0][cell(sx, sy)]);
        std::vector<satx::lit_t> goal;
        for (int t = 0; t <= T; ++t) goal.push_back(p[t][cell(gx, gy)]);
        e.add_clause(goal);
        for (int t = 0; t <= T; ++t)
            for (int c1 = 0; c1 < G * G; ++c1)
                for (int c2 = c1 + 1; c2 < G * G; ++c2)
                    e.add_unit(-satx::gates::and2(e, p[t][c1], p[t][c2]));
        for (int t = 1; t <= T; ++t)
            for (int x = 0; x < G; ++x)
                for (int y = 0; y < G; ++y) {
                    if (wall[y][x]) { e.add_unit(-p[t][cell(x, y)]); continue; }
                    std::vector<satx::lit_t> c = {-p[t][cell(x, y)]};
                    if (x > 0 && !wall[y][x - 1]) c.push_back(p[t - 1][cell(x - 1, y)]);
                    if (x + 1 < G && !wall[y][x + 1]) c.push_back(p[t - 1][cell(x + 1, y)]);
                    if (y > 0 && !wall[y - 1][x]) c.push_back(p[t - 1][cell(x, y - 1)]);
                    if (y + 1 < G && !wall[y + 1][x]) c.push_back(p[t - 1][cell(x, y + 1)]);
                    e.add_clause(c);
                }
    };

    // búsqueda binaria del mínimo T
    int lo = 1, hi = 14;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        satx::engine e;
        std::vector<std::vector<satx::lit_t>> p(mid + 1, std::vector<satx::lit_t>(G * G));
        build(e, mid, p);
        if (satx::solver::solve(e)) hi = mid;
        else lo = mid + 1;
    }
    std::printf("pasos mínimos (SLIME) = %d\n", lo);

    // ── oráculo BFS del host ────────────────────────────────────────────────
    std::vector<int> dist(G * G, -1);
    std::queue<int> q;
    dist[cell(sx, sy)] = 0;
    q.push(cell(sx, sy));
    while (!q.empty()) {
        const int c = q.front();
        q.pop();
        const int x = c % G, y = c / G;
        const int dd[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (auto const& [dx, dy] : dd) {
            const int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= G || ny < 0 || ny >= G || wall[ny][nx]) continue;
            if (dist[cell(nx, ny)] >= 0) continue;
            dist[cell(nx, ny)] = dist[c] + 1;
            q.push(cell(nx, ny));
        }
    }
    const int oracle = dist[cell(gx, gy)];
    std::printf("pasos mínimos (oráculo BFS) = %d %s\n", oracle, lo == oracle ? "(ok)" : "(FAIL)");
    if (lo != oracle) return EXIT_FAILURE;

    // camino con el mínimo: extraer y verificar
    satx::engine e;
    std::vector<std::vector<satx::lit_t>> p(lo + 1, std::vector<satx::lit_t>(G * G));
    build(e, lo, p);
    const auto sol = satx::solver::solve(e);
    if (!sol) return EXIT_FAILURE;
    std::printf("camino:");
    for (int t = 0; t <= lo; ++t)
        for (int x = 0; x < G; ++x)
            for (int y = 0; y < G; ++y)
                if (sol->get(p[t][cell(x, y)])) std::printf(" (%d,%d)", x, y);
    std::printf("\n");
    return EXIT_SUCCESS;
}
