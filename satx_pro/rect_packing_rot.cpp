// rect_packing_rot — Empaquetado 2D estilo mochila CON ROTACIÓN de 90°:
// ¿cuántos rectángulos aleatorios caben en un contenedor, pudiendo girar?
//
// Variante de examples/rect_packing.cpp: cada rectángulo i puede colocarse
// como está (w_i × h_i) o girado 90° (h_i × w_i). Se agrega un literal de
// rotación r_i y las dimensiones EFECTIVAS se construyen con un multiplexor
// bit a bit sobre los rieles NB de las constantes (sin circuitos: cada bit
// de salida es una constante o el propio r_i/¬r_i):
//   w'_i = r_i ? h_i : w_i     h'_i = r_i ? w_i : h_i
//
// Cada rectángulo i tiene literal de activación a_i y de rotación r_i:
//   · contención (solo si se coloca): a_i → 0 ≤ x_i ≤ W − max(w_i, h_i)
//                                              ∧ x_i + w'_i ≤ W
//                                              ∧ 0 ≤ y_i ≤ H − max(w_i, h_i)
//                                              ∧ y_i + h'_i ≤ H
//     La cota constante x ≤ W − max(w,h) evita la envoltura mod 2^W del
//     datapath (ADR-011): con ella, x + w' ≤ W ≤ max_NB(10) = 341, exacta.
//   · sin solapamiento (solo si ambos se colocan): a_i ∧ a_j → separación
//     con las dimensiones efectivas:
//     x_i + w'_i ≤ x_j ∨ x_j + w'_j ≤ x_i ∨ y_i + h'_i ≤ y_j ∨ y_j + h'_j ≤ y_i
//   · objetivo: maximizar Σ a_i — la mayor CANTIDAD de rectángulos que entra,
//     con SAT como oráculo (patrón de examples/optimize_sum.cpp y de
//     rect_packing.cpp): en cada vuelta se exige una solución con más piezas
//     hasta que la fórmula es UNSAT; ese es el óptimo.
//
// Al final se verifica la solución con una comprobación independiente en el
// host (con las dimensiones efectivas) y se escribe un JSON con los datos del
// problema (contenedor y piezas, con su columna inicial) y de la solución
// (posiciones empaquetadas, qué piezas quedaron fuera y cuáles giraron),
// listo para graficar con Python/matplotlib (examples/plot_packing_rot.py).
//
// Uso: rect_packing_rot [archivo.json] [N] [H] [W] [semilla]
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
  std::vector<C> xs, ys, weff, heff;   // posición y dimensiones efectivas
  std::vector<satx::lit_t> act, rot;   // a_i: ¿se coloca? r_i: ¿gira 90°?
  C total;                             // Σ a_i (0/1 por rectángulo)
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

// Multiplexor sel ? b : a entre dos CONSTANTES enteras, bit a bit sobre los
// rieles NB (sin circuitos: cada bit de salida es una constante o sel/¬sel).
C pick_const(satx::engine& e, satx::lit_t sel, std::int64_t a, std::int64_t b) {
  const auto ra = C{static_cast<double>(a), 0.0}.real_rail(e);
  const auto rb = C{static_cast<double>(b), 0.0}.real_rail(e);
  std::array<satx::lit_t, C::width> re{}, im{};
  im.fill(satx::core::false_lit);
  for (std::size_t k = 0; k < C::width; ++k) {
    const bool ba = ra[k] == satx::core::true_lit;
    const bool bb = rb[k] == satx::core::true_lit;
    if (ba == bb)
      re[k] = ba ? satx::core::true_lit : satx::core::false_lit;
    else
      re[k] = ba ? -sel : sel;
  }
  return C::from_nb_rails(e, re, im);
}

// Circuito SAT del problema: activación + rotación (dimensión efectiva) +
// contención + no solapamiento + objetivo (cantidad de piezas colocadas).
Encoding build(satx::engine& e, int W, int H, const std::vector<Rect>& rs) {
  const auto num = [](double v) { return C{v, 0.0}; };
  const int n = static_cast<int>(rs.size());

  Encoding enc;
  enc.xs.reserve(n);
  enc.ys.reserve(n);
  enc.weff.reserve(n);
  enc.heff.reserve(n);
  enc.act.reserve(n);
  enc.rot.reserve(n);
  for (int i = 0; i < n; ++i) {
    enc.xs.emplace_back(e);
    enc.ys.emplace_back(e);
    // solo coordenadas reales: parte imaginaria ≡ 0
    for (auto l : enc.xs[i].im_pattern()) e.add_unit(-l);
    for (auto l : enc.ys[i].im_pattern()) e.add_unit(-l);
    // rotación: dimensiones efectivas r_i ? (h × w) : (w × h)
    enc.rot.push_back(e.add_variable());
    enc.weff.push_back(pick_const(e, enc.rot.back(), rs[i].w, rs[i].h));
    enc.heff.push_back(pick_const(e, enc.rot.back(), rs[i].h, rs[i].w));
    // contención (solo si la pieza se coloca): a_i → cotas directas de la
    // variable (ver cabecera sobre la envoltura mod 2^W) + x + w' ≤ W
    const int mx = W - std::max(rs[i].w, rs[i].h);
    const int my = H - std::max(rs[i].w, rs[i].h);
    satx::lit_t c = satx::gates::and2(
        e, satx::le_lit(e, num(0), enc.xs[i]),
        satx::le_lit(e, enc.xs[i], num(mx)));
    c = satx::gates::and2(e, c, satx::le_lit(e, num(0), enc.ys[i]));
    c = satx::gates::and2(e, c, satx::le_lit(e, enc.ys[i], num(my)));
    c = satx::gates::and2(e, c, satx::le_lit(e, enc.xs[i] + enc.weff[i], num(W)));
    c = satx::gates::and2(e, c, satx::le_lit(e, enc.ys[i] + enc.heff[i], num(H)));
    enc.act.push_back(e.add_variable());
    e.add_clause({-enc.act.back(), c});
  }

  // no solapamiento (solo si ambas piezas se colocan): a_i ∧ a_j → separación
  // con las dimensiones efectivas (rotación incluida)
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const auto sep_x = satx::gates::or2(
          e, satx::le_lit(e, enc.xs[i] + enc.weff[i], enc.xs[j]),
          satx::le_lit(e, enc.xs[j] + enc.weff[j], enc.xs[i]));
      const auto sep_y = satx::gates::or2(
          e, satx::le_lit(e, enc.ys[i] + enc.heff[i], enc.ys[j]),
          satx::le_lit(e, enc.ys[j] + enc.heff[j], enc.ys[i]));
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

// Verificación independiente en el host: solo piezas colocadas; contención y
// sin solapamiento con las dimensiones EFECTIVAS (rotación aplicada), y
// cantidad alcanzada (Σ a_i == objetivo).
bool verify(int W, int H, const std::vector<Rect>& rs,
            const std::vector<std::pair<int, int>>& pos,
            const std::vector<bool>& placed, const std::vector<bool>& rot,
            int objetivo) {
  const int n = static_cast<int>(rs.size());
  int count = 0;
  for (int i = 0; i < n; ++i) {
    if (!placed[i]) continue;
    ++count;
    const int w = rot[i] ? rs[i].h : rs[i].w;
    const int h = rot[i] ? rs[i].w : rs[i].h;
    const int x = pos[i].first, y = pos[i].second;
    if (x < 0 || y < 0 || x + w > W || y + h > H) return false;
    for (int j = 0; j < n; ++j) {
      if (i == j || !placed[j]) continue;
      const int wj = rot[j] ? rs[j].h : rs[j].w;
      const int hj = rot[j] ? rs[j].w : rs[j].h;
      const int xj = pos[j].first, yj = pos[j].second;
      const bool sep_x = x >= xj + wj || xj >= x + w;
      const bool sep_y = y >= yj + hj || yj >= y + h;
      if (!sep_x && !sep_y) return false;
    }
  }
  return count == objetivo;
}

// JSON con los datos del problema (contenedor, piezas y columna inicial) y de
// la solución (posiciones empaquetadas, orientación y qué piezas se
// colocaron), para plot_packing_rot.py.
void write_json(std::ostream& out, int seed, int W, int H,
                const std::vector<Rect>& rs, const std::vector<int>& sx,
                const std::vector<int>& sy,
                const std::vector<std::pair<int, int>>& pos,
                const std::vector<bool>& placed, const std::vector<bool>& rot,
                int bbox_w, int bbox_h, int objetivo, long long variables,
                long long clausulas, int resoluciones) {
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
    const int w = rot[i] ? rs[i].h : rs[i].w;
    const int h = rot[i] ? rs[i].w : rs[i].h;
    out << "      {\"id\": " << i << ", \"x\": " << (placed[i] ? pos[i].first : 0)
        << ", \"y\": " << (placed[i] ? pos[i].second : 0)
        << ", \"w_eff\": " << w << ", \"h_eff\": " << h
        << ", \"rotado\": " << (rot[i] ? "true" : "false")
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
  const std::string out_path = argc > 1 ? argv[1] : "rect_packing_rot.json";
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
            << ", contenedor: " << W << 'x' << H
            << " (con rotacion de 90 grados)\n";

  std::vector<Rect> rs;
  gen_rects(n, W, H, rs);

  satx::engine e;
  const Encoding enc = build(e, W, H, rs);
  std::cout << "circuito: variables=" << e.variable_count()
            << ", clausulas=" << e.clause_count() << '\n';

  // Optimización con SAT como oráculo (patrón de rect_packing/optimize_sum):
  // en cada vuelta se pide Σ a_i > best añadiendo una cláusula unitaria;
  // cuando la fórmula es UNSAT, best es la máxima cantidad que entra.
  // Manual §13.7: para circuitos CBE grandes conviene la heurística CHB.
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

  // Decodificación exacta (F = 0 → componentes enteras) y verificación con
  // las dimensiones efectivas (rotación aplicada).
  std::vector<std::pair<int, int>> pos(n);
  std::vector<bool> placed(n), rot(n);
  for (int i = 0; i < n; ++i) {
    pos[i] = {static_cast<int>(enc.xs[i].value_raw(mejor).real()),
              static_cast<int>(enc.ys[i].value_raw(mejor).real())};
    placed[i] = mejor.get(enc.act[i]);
    rot[i] = mejor.get(enc.rot[i]);
  }
  if (!verify(W, H, rs, pos, placed, rot, best)) {
    std::cerr << "error: verificacion del host FALLIDA\n";
    return EXIT_FAILURE;
  }
  std::cout << "verificacion del host: OK\n";

  // Datos para graficar: columna de piezas (vista del problema) y caja de la
  // solución (solo piezas colocadas, dimensiones efectivas).
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
    const int w = rot[i] ? rs[i].h : rs[i].w;
    const int h = rot[i] ? rs[i].w : rs[i].h;
    bbox_w = std::max(bbox_w, pos[i].first + w);
    bbox_h = std::max(bbox_h, pos[i].second + h);
  }

  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "error: no se pudo abrir '" << out_path << "'\n";
    return EXIT_FAILURE;
  }
  write_json(out, seed, W, H, rs, sx, sy, pos, placed, rot, bbox_w, bbox_h,
             best, static_cast<long long>(e.variable_count()),
             static_cast<long long>(e.clause_count()), resoluciones);
  out.close();
  std::cout << "JSON: " << out_path << '\n';
  return EXIT_SUCCESS;
}
