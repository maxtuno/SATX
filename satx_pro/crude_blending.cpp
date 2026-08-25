// crude_blending — catálogo 4.3: mezcla de crudos con índices no lineales,
// kernel SLIME (discretización exacta con la aritmética CBE) + BASILISK.
//
// Fracciones x_i en pasos de 1/N (N = 10) con Σ x_i = 10. La viscosidad se
// modela con un término bilineal x_0·x_1 (propiedad no aditiva), que CBE
// multiplica exactamente; la densidad es lineal. Se exigen bandas:
//
//   densidad:  5·x_0 + 6·x_1 + 7·x_2 ∈ [59, 65]
//   viscosidad: 4·x_0 + 6·x_1 + 8·x_2 + x_0·x_1 ∈ [70, 74]
//
// BASILISK cuenta las recetas factibles (espacio de diseño); se compara con
// la enumeración de las 66 combinaciones del host.

#include <satx/satx.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int N = 10;  // pasos de 1/N

    using I = satx::integer<8>;  // fracciones 0..10 y bandas 0..74 (rango -128..127)

    satx::engine e;
    I x0{e}, x1{e}, x2{e};
    for (const I* x : {&x0, &x1, &x2}) {
        e.add_unit(satx::le_lit(e, I{0}, *x));
        e.add_unit(satx::le_lit(e, *x, I{N}));
    }
    // Σ x_i = 10
    e.add_unit(satx::eq_lit(e, x0 + x1 + x2, I{10}));
    // densidad (lineal)
    e.add_unit(satx::ge_lit(e, I{5} * x0 + I{6} * x1 + I{7} * x2, I{59}));
    e.add_unit(satx::le_lit(e, I{5} * x0 + I{6} * x1 + I{7} * x2, I{65}));
    // viscosidad (bilineal: el producto x0·x1 es exacto en CBE)
    const auto visc = I{4} * x0 + I{6} * x1 + I{8} * x2 + x0 * x1;
    e.add_unit(satx::ge_lit(e, visc, I{70}));
    e.add_unit(satx::le_lit(e, visc, I{74}));

    std::printf("mezcla de crudos: paso 1/%d (variables %zu, cláusulas %zu)\n", N,
                e.variable_count(), e.clause_count());
    const auto sol = satx::solver::solve(e);
    if (!sol) {
        std::printf("VERIFICACIÓN FALLIDA: se esperaba SAT\n");
        return EXIT_FAILURE;
    }
    const int v0 = static_cast<int>(x0.value(*sol));
    const int v1 = static_cast<int>(x1.value(*sol));
    const int v2 = static_cast<int>(x2.value(*sol));
    std::printf("receta: x0=%d%% x1=%d%% x2=%d%%\n", 10 * v0, 10 * v1, 10 * v2);
    const int D = 5 * v0 + 6 * v1 + 7 * v2;
    const int P = 4 * v0 + 6 * v1 + 8 * v2 + v0 * v1;
    std::printf("densidad=%d, viscosidad=%d\n", D, P);
    if (D < 59 || D > 65 || P < 70 || P > 74) {
        std::printf("VERIFICACIÓN FALLIDA: bandas\n");
        return EXIT_FAILURE;
    }

    // ── conteo de recetas factibles ─────────────────────────────────────────
    const auto cnt = satx::solver::basilisk::count(e);
    std::printf("recetas factibles (BASILISK) = %s\n", cnt.value().c_str());

    int oracle = 0;
    for (int a = 0; a <= N; ++a)
        for (int b = 0; b <= N - a; ++b) {
            const int c = N - a - b;
            const int dd = 5 * a + 6 * b + 7 * c;
            const int pp = 4 * a + 6 * b + 8 * c + a * b;
            if (dd >= 59 && dd <= 65 && pp >= 70 && pp <= 74) ++oracle;
        }
    std::printf("recetas factibles (oráculo) = %d %s\n", oracle,
                cnt.as_double() == static_cast<double>(oracle) ? "(ok)" : "(FAIL)");
    if (cnt.as_double() != static_cast<double>(oracle)) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
