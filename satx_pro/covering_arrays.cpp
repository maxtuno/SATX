// covering_arrays — catálogo 5.3: diseño de experimentos — arrays de
// cobertura, kernel SLIME + BASILISK.
//
// (a) Fuerza 2 con 4 parámetros × 2 valores: buscar el mínimo de filas R tal
//     que toda combinación de pares (parámetro, valor) aparezca en alguna
//     fila (cada fila es un test de configuración).
// (b) BASILISK cuenta los diseños admisibles de un caso pequeño
//     (2 parámetros × 2 valores, R = 3 filas) — el oráculo lo verifica por
//     enumeración.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    // ── (a) mínimo de filas para fuerza 2 ───────────────────────────────────
    constexpr int NP = 4, NV = 2, S2 = 2;
    const int Rmax = 8;

    const auto build = [](satx::engine& e, int R,
                          std::vector<std::vector<std::vector<satx::lit_t>>>& x) {
        for (int r = 0; r < R; ++r)
            for (int p = 0; p < NP; ++p) {
                x[r][p][0] = e.add_variable();
                x[r][p][1] = e.add_variable();
                // exactamente un valor por parámetro y fila
                e.add_unit(-satx::gates::and2(e, x[r][p][0], x[r][p][1]));
                e.add_clause({x[r][p][0], x[r][p][1]});
            }
        // cobertura de pares (fuerza 2)
        for (int p1 = 0; p1 < NP; ++p1)
            for (int p2 = p1 + 1; p2 < NP; ++p2)
                for (int v1 = 0; v1 < NV; ++v1)
                    for (int v2 = 0; v2 < NV; ++v2) {
                        std::vector<satx::lit_t> c;
                        for (int r = 0; r < R; ++r)
                            c.push_back(satx::gates::and2(e, x[r][p1][v1], x[r][p2][v2]));
                        e.add_clause(c);
                    }
    };

    const auto verify = [](int R, satx::solver::model const& m,
                           std::vector<std::vector<std::vector<satx::lit_t>>> const& x) {
        for (int p1 = 0; p1 < NP; ++p1)
            for (int p2 = p1 + 1; p2 < NP; ++p2)
                for (int v1 = 0; v1 < NV; ++v1)
                    for (int v2 = 0; v2 < NV; ++v2) {
                        bool covered = false;
                        for (int r = 0; r < R && !covered; ++r)
                            covered = m.get(x[r][p1][v1]) && m.get(x[r][p2][v2]);
                        if (!covered) return false;
                    }
        return true;
    };

    int Rmin = -1;
    for (int R = 4; R <= Rmax; ++R) {
        satx::engine e;
        std::vector<std::vector<std::vector<satx::lit_t>>> x(
            R, std::vector<std::vector<satx::lit_t>>(NP, std::vector<satx::lit_t>(NV)));
        build(e, R, x);
        const auto sol = satx::solver::solve(e);
        if (!sol) continue;
        if (!verify(R, *sol, x)) {
            std::printf("VERIFICACIÓN FALLIDA: cobertura de R=%d\n", R);
            return EXIT_FAILURE;
        }
        Rmin = R;
        std::printf("(a) fuerza 2, 4×2: mínimo de filas = %d (SAT)\n", R);
        for (int r = 0; r < R; ++r) {
            std::printf("    test %d:", r);
            for (int p = 0; p < NP; ++p)
                for (int v = 0; v < NV; ++v)
                    if (sol->get(x[r][p][v])) std::printf("  p%d=%d", p, v);
            std::printf("\n");
        }
        break;
    }
    if (Rmin < 0) {
        std::printf("VERIFICACIÓN FALLIDA: sin cobertura hasta R=%d\n", Rmax);
        return EXIT_FAILURE;
    }

    // ── (b) conteo de diseños con BASILISK ──────────────────────────────────
    // 3 parámetros × 2 valores, fuerza 2, R = 4 filas: cada fila cubre 3
    // pares; existen 8^4 = 4096 asignaciones de filas y el OA(4,3,2) asegura
    // diseños válidos. BASILISK cuenta los diseños; el oráculo los enumera.
    {
        constexpr int R = 4, P = 3, V = 2;
        satx::engine e;
        std::vector<std::vector<std::vector<satx::lit_t>>> x(
            R, std::vector<std::vector<satx::lit_t>>(P, std::vector<satx::lit_t>(V)));
        for (int r = 0; r < R; ++r)
            for (int p = 0; p < P; ++p) {
                x[r][p][0] = e.add_variable();
                x[r][p][1] = e.add_variable();
                e.add_unit(-satx::gates::and2(e, x[r][p][0], x[r][p][1]));
                e.add_clause({x[r][p][0], x[r][p][1]});
            }
        for (int p1 = 0; p1 < P; ++p1)
            for (int p2 = p1 + 1; p2 < P; ++p2)
                for (int v1 = 0; v1 < V; ++v1)
                    for (int v2 = 0; v2 < V; ++v2) {
                        std::vector<satx::lit_t> c;
                        for (int r = 0; r < R; ++r)
                            c.push_back(satx::gates::and2(e, x[r][p1][v1], x[r][p2][v2]));
                        e.add_clause(c);
                    }
        const auto cnt = satx::solver::basilisk::count(e);
        std::printf("(b) diseños admisibles (BASILISK, 3×2, fuerza 2, R=4) = %s\n",
                    cnt.value().c_str());

        // oráculo: 8^4 filas posibles (v0,v1,v2 por fila, codificado 0..7)
        int oracle = 0;
        for (std::size_t code = 0; code < 4096; ++code) {
            std::size_t k = code;
            int rowv[R];
            for (int r = 0; r < R; ++r) {
                rowv[r] = static_cast<int>(k & 7u);
                k >>= 3;
            }
            bool ok = true;
            for (int p1 = 0; p1 < P && ok; ++p1)
                for (int p2 = p1 + 1; p2 < P && ok; ++p2)
                    for (int v1 = 0; v1 < V && ok; ++v1)
                        for (int v2 = 0; v2 < V && ok; ++v2) {
                            bool cov = false;
                            for (int r = 0; r < R && !cov; ++r)
                                cov = ((rowv[r] >> p1) & 1) == v1 && ((rowv[r] >> p2) & 1) == v2;
                            if (!cov) ok = false;
                        }
            if (ok) ++oracle;
        }
        std::printf("(b) diseños admisibles (oráculo) = %d %s\n", oracle,
                    cnt.as_double() == static_cast<double>(oracle) ? "(ok)" : "(FAIL)");
        if (cnt.as_double() != static_cast<double>(oracle)) return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
