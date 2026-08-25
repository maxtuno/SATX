// procedural_map — catálogo 5.4: generación procedural de mapas, kernel
// SLIME.
//
// Mapa 4×4 con casillas suelo/muro; el jugador camina del inicio a la meta
// en T pasos; la puerta solo se puede pisar después de haber recogido la
// llave. Las celdas libres (suelo) son las que el camino necesita — el
// solver diseña el mapa y el camino a la vez.
//
// Verificación: extracción del camino en el host (adyacencia, llave antes
// que puerta, meta alcanzada).

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int G = 4, T = 9;  // rejilla 4×4, pasos 0..T
    const int sx = 0, sy = 0, gx = 3, gy = 3;  // inicio, meta
    const int kx = 1, ky = 1, dx = 2, dy = 2;  // llave, puerta

    satx::engine e;
    // p[t][y*G+x]: el jugador está en (x,y) en el paso t
    std::vector<std::vector<satx::lit_t>> p(T + 1, std::vector<satx::lit_t>(G * G));
    // f[y*G+x]: la celda es suelo
    std::vector<satx::lit_t> f(G * G);
    for (int t = 0; t <= T; ++t)
        for (int c = 0; c < G * G; ++c) p[t][c] = e.add_variable();
    for (int c = 0; c < G * G; ++c) f[c] = e.add_variable();

    const auto cell = [&](int x, int y) { return y * G + x; };

    // inicio en el paso 0; meta en algún paso
    e.add_unit(p[0][cell(sx, sy)]);
    {
        std::vector<satx::lit_t> c;
        for (int t = 0; t <= T; ++t) c.push_back(p[t][cell(gx, gy)]);
        e.add_clause(c);
    }
    // el camino debe pasar por la puerta (si no, la precedencia es vacua)
    {
        std::vector<satx::lit_t> c;
        for (int t = 0; t <= T; ++t) c.push_back(p[t][cell(dx, dy)]);
        e.add_clause(c);
    }
    // a lo sumo una celda por paso
    for (int t = 0; t <= T; ++t)
        for (int c1 = 0; c1 < G * G; ++c1)
            for (int c2 = c1 + 1; c2 < G * G; ++c2)
                e.add_unit(-satx::gates::and2(e, p[t][c1], p[t][c2]));
    // transición: p[t][c] → ∨_{adyacente} p[t-1][c']
    for (int t = 1; t <= T; ++t)
        for (int x = 0; x < G; ++x)
            for (int y = 0; y < G; ++y) {
                std::vector<satx::lit_t> c = {-p[t][cell(x, y)]};
                if (x > 0) c.push_back(p[t - 1][cell(x - 1, y)]);
                if (x + 1 < G) c.push_back(p[t - 1][cell(x + 1, y)]);
                if (y > 0) c.push_back(p[t - 1][cell(x, y - 1)]);
                if (y + 1 < G) c.push_back(p[t - 1][cell(x, y + 1)]);
                e.add_clause(c);
            }
    // p[t][c] → f[c]
    for (int t = 0; t <= T; ++t)
        for (int c = 0; c < G * G; ++c) e.add_unit(-satx::gates::and2(e, p[t][c], -f[c]));
    // puerta solo después de la llave: collected[t] = ∨_{t'≤t} p[t'][llave]
    std::vector<satx::lit_t> collected(T + 1, satx::core::false_lit);
    for (int t = 0; t <= T; ++t) {
        collected[t] = t == 0 ? p[t][cell(kx, ky)]
                              : satx::gates::or2(e, collected[t - 1], p[t][cell(kx, ky)]);
        if (t > 0) e.add_unit(-satx::gates::and2(e, p[t][cell(dx, dy)], -collected[t - 1]));
    }
    // la puerta y la llave existen (son suelo)
    e.add_unit(f[cell(kx, ky)]);
    e.add_unit(f[cell(dx, dy)]);

    std::printf("mapa procedural: %d×%d, %d pasos (variables %zu, cláusulas %zu)\n", G, G, T,
                e.variable_count(), e.clause_count());

    const auto sol = satx::solver::solve(e);
    if (!sol) {
        std::printf("VERIFICACIÓN FALLIDA: se esperaba SAT\n");
        return EXIT_FAILURE;
    }

    // ── extracción y verificación del camino ────────────────────────────────
    std::vector<std::pair<int, int>> path;
    for (int t = 0; t <= T; ++t)
        for (int x = 0; x < G; ++x)
            for (int y = 0; y < G; ++y)
                if (sol->get(p[t][cell(x, y)])) path.emplace_back(x, y);

    std::printf("camino (%zu pasos):", path.size());
    for (auto const& [x, y] : path) std::printf(" (%d,%d)", x, y);
    std::printf("\n");

    if (path.empty() || path[0] != std::pair{0, 0}) {
        std::printf("VERIFICACIÓN FALLIDA: inicio\n");
        return EXIT_FAILURE;
    }
    int key_step = -1, door_step = -1, goal_step = -1;
    for (std::size_t t = 0; t < path.size(); ++t) {
        const auto& [x, y] = path[t];
        if (x == kx && y == ky && key_step < 0) key_step = static_cast<int>(t);
        if (x == dx && y == dy && door_step < 0) door_step = static_cast<int>(t);
        if (x == gx && y == gy && goal_step < 0) goal_step = static_cast<int>(t);
        if (t > 0) {
            const auto& [px, py] = path[t - 1];
            if (std::abs(x - px) + std::abs(y - py) != 1) {
                std::printf("VERIFICACIÓN FALLIDA: salto (%d,%d)→(%d,%d)\n", px, py, x, y);
                return EXIT_FAILURE;
            }
        }
    }
    if (goal_step < 0 || key_step < 0 || door_step < 0) {
        std::printf("VERIFICACIÓN FALLIDA: llave (%d), puerta (%d) o meta (%d) no visitadas\n",
                    key_step, door_step, goal_step);
        return EXIT_FAILURE;
    }
    if (door_step <= key_step) {
        std::printf("VERIFICACIÓN FALLIDA: la puerta se cruza antes de la llave\n");
        return EXIT_FAILURE;
    }
    std::printf("llave en paso %d, puerta en paso %d, meta en paso %d\n", key_step, door_step,
                goal_step);
    std::printf("verificación del host: OK\n");
    return EXIT_SUCCESS;
}
