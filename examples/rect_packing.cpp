// rect_packing — Empaquetado 2D estilo mochila: ¿cuántos rectángulos
// aleatorios caben en un contenedor dado?
//
// Se genera una instancia CUALQUIERA (sin elegir una "bonita"): N rectángulos
// con dimensiones aleatorias derivadas del contenedor. El contenedor W×H, la
// cantidad N y la semilla se reciben por línea de comandos.
//
// Cada rectángulo i tiene un literal de activación a_i (¿se coloca o no?):
//   · contención (solo si se coloca):  a_i → 0 ≤ x_i ≤ W − w_i
//                                              ∧ 0 ≤ y_i ≤ H − h_i
//   · sin solapamiento (solo si ambos se colocan): a_i ∧ a_j → separación
//     x_i + w_i ≤ x_j ∨ x_j + w_j ≤ x_i ∨ y_i + h_i ≤ y_j ∨ y_j + h_j ≤ y_i
//   · objetivo: maximizar Σ a_i — la mayor CANTIDAD de rectángulos que entra,
//     análogo al problema de la mochila — con SAT como oráculo (patrón de
//     examples/optimize_sum.cpp): en cada vuelta se exige una solución con más
//     piezas (cláusula unitaria lt_lit(num(best), total)) hasta que la
//     fórmula es UNSAT; ese es el óptimo.
//
// Nota de envoltura: el datapath suma mod 2^W (ADR-011), así que x + w con x
// grande "rebosa" a un valor negativo. Por eso la contención acota la
// VARIABLE directamente (0 ≤ x ≤ W − w); con esa cota, x + w ≤ W es exacta
// (nunca envuelve: ≤ W < max_NB(10) = 341).
//
// Al final se verifica la solución con una comprobación independiente en el
// host y se escribe un JSON con los datos del problema (contenedor y piezas,
// con su columna inicial para la vista del problema) y de la solución
// (posiciones empaquetadas y qué piezas quedaron fuera), listo para graficar
// con Python/matplotlib (examples/plot_packing.py).
//
// Uso: rect_packing [archivo.json] [N] [H] [W] [semilla]
//   N: cantidad de rectángulos (por defecto 8)
//   H, W: tamaño del espacio (por defecto 10 × 14)
//   El empaquetado 2D es NP-completo: no abuses de N ni del contenedor.

#include <satx/satx.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using C = satx::complex<10, 0>;  // coordenadas enteras (F = 0); caja NB [−682, 341]

struct Rect {
  int w, h;
};

struct Encoding {
  std::vector<C> xs, ys;            // posición de cada rectángulo
  std::vector<satx::lit_t> act;     // a_i: ¿se coloca el rectángulo i?
  C total;                          // Σ a_i (0/1 por rectángulo)
};

std::mt19937 rng;

int ri(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }

// Instancia cualquiera: N rectángulos con dimensiones aleatorias entre 2 y
// min(W, H)/2 (como mínimo 2). Sin filtros de "bonita": si algunos no caben,
// la optimización los deja fuera (mochila).
void gen_rects(int n, int W, int H, std::vector<Rect>& rs) {
  const int s_max = std::max(2, std::min(W, H) / 2);
  rs.clear();
  rs.reserve(n);
  for (int i = 0; i < n; ++i) rs.push_back({ri(2, s_max), ri(2, s_max)});
}

// Circuito SAT del problema: activación + contención + no solapamiento +
// objetivo (cantidad de piezas colocadas).
Encoding build(satx::engine& e, int W, int H, const std::vector<Rect>& rs) {
  const auto num = [](double v) { return C{v, 0.0}; };
  const int n = static_cast<int>(rs.size());

  Encoding enc;
  enc.xs.reserve(n);
  enc.ys.reserve(n);
  enc.act.reserve(n);
  for (int i = 0; i < n; ++i) {
    enc.xs.emplace_back(e);
    enc.ys.emplace_back(e);
    // solo coordenadas reales: parte imaginaria ≡ 0
    for (auto l : enc.xs[i].im_pattern()) e.add_unit(-l);
    for (auto l : enc.ys[i].im_pattern()) e.add_unit(-l);
    // contención (solo si la pieza se coloca): a_i → cotas directas de la
    // variable (ver cabecera sobre la envoltura mod 2^W)
    satx::lit_t c = satx::gates::and2(
        e, satx::le_lit(e, num(0), enc.xs[i]),
        satx::le_lit(e, enc.xs[i], num(W - rs[i].w)));
    c = satx::gates::and2(e, c, satx::le_lit(e, num(0), enc.ys[i]));
    c = satx::gates::and2(e, c, satx::le_lit(e, enc.ys[i], num(H - rs[i].h)));
    c = satx::gates::and2(e, c, satx::le_lit(e, enc.xs[i] + num(rs[i].w), num(W)));  // redundante
    c = satx::gates::and2(e, c, satx::le_lit(e, enc.ys[i] + num(rs[i].h), num(H)));  // redundante
    enc.act.push_back(e.add_variable());
    e.add_clause({-enc.act.back(), c});
  }

  // no solapamiento (solo si ambas piezas se colocan): a_i ∧ a_j → separación
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const auto sep_x = satx::gates::or2(
          e, satx::le_lit(e, enc.xs[i] + num(rs[i].w), enc.xs[j]),
          satx::le_lit(e, enc.xs[j] + num(rs[j].w), enc.xs[i]));
      const auto sep_y = satx::gates::or2(
          e, satx::le_lit(e, enc.ys[i] + num(rs[i].h), enc.ys[j]),
          satx::le_lit(e, enc.ys[j] + num(rs[j].h), enc.ys[i]));
      const auto sep = satx::gates::or2(e, sep_x, sep_y);
      e.add_clause({-enc.act[i], -enc.act[j], sep});
    }
  }

  // objetivo: total = Σ a_i como complejo 0/1 (literal a_i en el bit 0 del
  // riel real); la suma no envuelve: ≤ N ≤ 15 < 341
  enc.total = num(0);
  for (int i = 0; i < n; ++i) {
    std::array<satx::lit_t, C::width> re{};
    std::array<satx::lit_t, C::width> im{};
    re.fill(satx::core::false_lit);
    im.fill(satx::core::false_lit);
    re[0] = enc.act[i];
    enc.total = enc.total + C::from_nb_rails(e, re, im);
  }
  return enc;
}

// Verificación independiente en el host: solo piezas colocadas; contención,
// sin solapamiento y cantidad alcanzada (Σ a_i == objetivo).
bool verify(int W, int H, const std::vector<Rect>& rs,
            const std::vector<std::pair<int, int>>& pos,
            const std::vector<bool>& placed, int objetivo) {
  const int n = static_cast<int>(rs.size());
  int count = 0;
  for (int i = 0; i < n; ++i) {
    if (!placed[i]) continue;
    ++count;
    const int x = pos[i].first, y = pos[i].second;
    if (x < 0 || y < 0 || x + rs[i].w > W || y + rs[i].h > H) return false;
    for (int j = 0; j < n; ++j) {
      if (i == j || !placed[j]) continue;
      const int xj = pos[j].first, yj = pos[j].second;
      const bool sep_x = x >= xj + rs[j].w || xj >= x + rs[i].w;
      const bool sep_y = y >= yj + rs[j].h || yj >= y + rs[i].h;
      if (!sep_x && !sep_y) return false;
    }
  }
  return count == objetivo;
}

// JSON con los datos del problema (contenedor, piezas y columna inicial) y de
// la solución (posiciones empaquetadas y qué piezas se colocaron), para
// plot_packing.py.
void write_json(std::ostream& out, int seed, int W, int H,
                const std::vector<Rect>& rs, const std::vector<int>& sx,
                const std::vector<int>& sy,
                const std::vector<std::pair<int, int>>& pos,
                const std::vector<bool>& placed, int bbox_w, int bbox_h,
                int objetivo, long long variables, long long clausulas,
                int resoluciones) {
  const int n = static_cast<int>(rs.size());
  int area_usada = 0;
  for (int i = 0; i < n; ++i)
    if (placed[i]) area_usada += rs[i].w * rs[i].h;
  out << "{\n";
  out << "  \"semilla\": " << seed << ",\n";
  out << "  \"contenedor\": {\"w\": " << W << ", \"h\": " << H << "},\n";
  out << "  \"rectangulos\": [\n";
  for (int i = 0; i < n; ++i) {
    out << "    {\"id\": " << i << ", \"w\": " << rs[i].w
        << ", \"h\": " << rs[i].h << ", \"sx\": " << sx[i]
        << ", \"sy\": " << sy[i] << "}";
    out << (i + 1 < n ? "," : "") << "\n";
  }
  out << "  ],\n";
  out << "  \"solucion\": {\n";
  out << "    \"colocados\": [\n";
  for (int i = 0; i < n; ++i) {
    out << "      {\"id\": " << i << ", \"x\": " << (placed[i] ? pos[i].first : 0)
        << ", \"y\": " << (placed[i] ? pos[i].second : 0)
        << ", \"colocado\": " << (placed[i] ? "true" : "false") << "}";
    out << (i + 1 < n ? "," : "") << "\n";
  }
  out << "    ],\n";
  out << "    \"caja\": {\"w\": " << bbox_w << ", \"h\": " << bbox_h << "},\n";
  out << "    \"area_usada\": " << area_usada << ",\n";
  out << "    \"area_contenedor\": " << W * H << ",\n";
  out << "    \"porcentaje_usado\": " << (100.0 * area_usada / (W * H)) << ",\n";
  out << "    \"objetivo\": " << objetivo << ",\n";
  out << "    \"variables\": " << variables << ",\n";
  out << "    \"clausulas\": " << clausulas << ",\n";
  out << "    \"resoluciones\": " << resoluciones << "\n";
  out << "  }\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string out_path = argc > 1 ? argv[1] : "rect_packing.json";
  int n = 8, H = 10, W = 14;
  std::uint32_t seed = std::random_device{}();
  if (argc > 2) n = std::max(1, std::stoi(argv[2]));
  if (argc > 3) H = std::max(2, std::stoi(argv[3]));
  if (argc > 4) W = std::max(2, std::stoi(argv[4]));
  if (argc > 5) seed = static_cast<std::uint32_t>(std::stoul(argv[5]));
  if (n >= 12 || static_cast<long long>(W) * H > 250)
    std::cout << "aviso: el empaquetado 2D es NP-completo; puede tardar.\n";
  rng.seed(seed);
  std::cout << "semilla: " << seed << ", rectangulos: " << n
            << ", contenedor: " << W << 'x' << H << '\n';

  std::vector<Rect> rs;
  gen_rects(n, W, H, rs);

  satx::engine e;
  const Encoding enc = build(e, W, H, rs);
  std::cout << "circuito: variables=" << e.variable_count()
            << ", clausulas=" << e.clause_count() << '\n';

  // Optimización con SAT como oráculo (patrón de optimize_sum): en cada
  // vuelta se pide Σ a_i > best añadiendo una cláusula unitaria; cuando la
  // fórmula es UNSAT, best es la máxima cantidad que entra. Manual §13.7:
  // para circuitos CBE grandes conviene la heurística CHB.
  const auto num = [](double v) { return C{v, 0.0}; };
  satx::solver::options opt;
  opt.heuristic_mode = 1;  // CHB (manual §13.7)
  int best = -1, resoluciones = 0;
  satx::solver::model mejor;
  for (;;) {
    auto m = satx::solver::solve(e, opt);
    ++resoluciones;
    if (!m) break;
    mejor = *m;
    const int v = static_cast<int>(enc.total.value_raw(*m).real());
    if (v <= best) {
      std::cerr << "error: la resolucion no mejoro el objetivo\n";
      return EXIT_FAILURE;
    }
    best = v;
    if (v >= n) break;  // ya caben todas: no hay más que pedir
    e.add_unit(satx::lt_lit(e, num(best), enc.total));  // la próxima debe tener una más
  }
  std::cout << "optimo: caben " << best << " de " << n
            << " rectangulos (resoluciones: " << resoluciones << ")\n";

  // Decodificación exacta (F = 0 → componentes enteras) y verificación.
  std::vector<std::pair<int, int>> pos(n);
  std::vector<bool> placed(n);
  for (int i = 0; i < n; ++i) {
    pos[i] = {static_cast<int>(enc.xs[i].value_raw(mejor).real()),
              static_cast<int>(enc.ys[i].value_raw(mejor).real())};
    placed[i] = mejor.get(enc.act[i]);
  }
  if (!verify(W, H, rs, pos, placed, best)) {
    std::cerr << "error: verificacion del host FALLIDA\n";
    return EXIT_FAILURE;
  }
  std::cout << "verificacion del host: OK\n";

  // Datos para graficar: columna de piezas (vista del problema) y caja de la
  // solución (solo piezas colocadas).
  std::vector<int> sx(n), sy(n);
  int qx = 0, qy = H + 2;
  for (int i = 0; i < n; ++i) {
    if (qx + rs[i].w > W) {
      qx = 0;
      qy += 6;
    }
    sx[i] = qx;
    sy[i] = qy;
    qx += rs[i].w + 1;
  }
  int bbox_w = 0, bbox_h = 0;
  for (int i = 0; i < n; ++i) {
    if (!placed[i]) continue;
    bbox_w = std::max(bbox_w, pos[i].first + rs[i].w);
    bbox_h = std::max(bbox_h, pos[i].second + rs[i].h);
  }

  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "error: no se pudo abrir '" << out_path << "'\n";
    return EXIT_FAILURE;
  }
  write_json(out, seed, W, H, rs, sx, sy, pos, placed, bbox_w, bbox_h, best,
             static_cast<long long>(e.variable_count()),
             static_cast<long long>(e.clause_count()), resoluciones);
  out.close();
  std::cout << "JSON: " << out_path << '\n';
  return EXIT_SUCCESS;
}
