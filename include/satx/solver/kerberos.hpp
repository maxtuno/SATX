#pragma once
// satx::solver — puente sobre el kernel SAT de Kerberos (CDCL SLIME)
// ADR-005 (revisado): la librería resuelve con Kerberos, no con un DPLL propio.
//
// Extensiones (ADR-011): opciones del kernel (espejo de SlimeSatOptions),
// estadísticas (SlimeSatStats), suposiciones (SAT bajo suposiciones) y
// sesiones incrementales (un solo handle, varias llamadas a solve).
// Nota de semántica de stats (slime.c): clauses/learnt son instantáneas
// absolutas (y NO incluyen las cláusulas unitarias, que SLIME propaga sin
// almacenar); el resto de contadores son deltas por llamada a solve.

#include "../core/engine.hpp"
#include "model.hpp"

#include <cstddef>
#include <expected>
#include <initializer_list>
#include <memory>
#include <span>

namespace satx::solver {

enum class result { sat, unsat };

// Opciones del kernel CDCL SLIME. Los valores por defecto coinciden con
// slime_sat_options_default de slime.c (VSIDS, sin MAB, mabc=4.0, sin HESS,
// covertrace ON, ct_lbd_max=6, ct_maxlen=12, ct_max_cubes=40000,
// ct_escape_rounds=4, ct_probe_restarts=4; simplificación raíz, chrono y
// sondeo de literales fallidos ON; inprocessing periódico OFF).
struct options {
  int heuristic_mode = 0;    // 0 = VSIDS, 1 = CHB
  int use_mab = 0;
  double mabc = 4.0;
  int use_hess = 0;
  int use_ct = 1;
  int ct_lbd_max = 6;
  int ct_maxlen = 12;
  int ct_max_cubes = 40000;
  int ct_buddy_merge = 0;
  int ct_escape_rounds = 4;
  int ct_probe_restarts = 4;
  int use_simplify = 1;      // preprocesamiento raíz (BVE, subsumción, etc.)
  int use_bve = 0;           // eliminación acotada de variables (experimental)
  int use_chrono = 1;        // backtracking cronológico
  int use_inprocess = 0;     // simplificación periódica durante la búsqueda
  int use_probe = 1;         // sondeo de literales fallidos (bounded)
};

// Estadísticas del kernel (espejo de SlimeSatStats).
struct stats {
  long long clauses = 0;      // instantánea absoluta
  long long learnt = 0;       // instantánea absoluta
  long long conflicts = 0;    // delta por llamada
  long long decisions = 0;
  long long propagations = 0;
  long long restarts = 0;
  long long hess_calls = 0;
  long long hess_sat_hits = 0;
  long long ct_added = 0;
  long long ct_merged = 0;
  long long ct_escaped = 0;
  long long ct_probe_added = 0;
};

// Resuelve la fórmula del engine con el backend SLIME de Kerberos (C API
// embebida, sin ficheros intermedios). budget: advisory; el puente embebido de
// SLIME no impone límite de conflictos en esta etapa — ADR-009.
[[nodiscard]] std::expected<model, result> solve(engine const& e, std::size_t budget = 0);

// Resuelve con opciones explícitas del kernel y, opcionalmente, recolecta
// estadísticas.
[[nodiscard]] std::expected<model, result> solve(engine const& e, options const& opt,
                                                 stats* out = nullptr);

// Resuelve bajo suposiciones (literales DIMACS 1..nvars; se revierten tras la
// llamada). Literales inválidos (0, INT32_MIN, fuera de [1, nvars]) →
// std::invalid_argument.
[[nodiscard]] std::expected<model, result> solve(engine const& e,
                                                 std::span<core::lit_t const> assumptions,
                                                 options const& opt = {}, stats* out = nullptr);
[[nodiscard]] std::expected<model, result> solve(engine const& e,
                                                 std::initializer_list<core::lit_t> assumptions,
                                                 options const& opt = {}, stats* out = nullptr);

// Sesión incremental: crea UNA vez el handle del kernel y reutiliza la base de
// cláusulas (con las cláusulas aprendidas) entre llamadas a solve con
// suposiciones distintas. RAII: destruye el handle al salir de alcance.
// Si la fórmula del engine ya es UNSAT en la construcción, el constructor
// lanza std::runtime_error (para ese caso usa solve(), que devuelve
// result::unsat).
class session {
public:
  explicit session(engine const& e, options const& opt = {});

  session(session const&) = delete;
  session& operator=(session const&) = delete;
  session(session&&) noexcept;
  session& operator=(session&&) noexcept;

  ~session();

  // Resuelve bajo suposiciones; devuelve el modelo (válido hasta la siguiente
  // llamada o la destrucción de la sesión).
  [[nodiscard]] std::expected<model, result> solve(
      std::span<core::lit_t const> assumptions = {}, stats* out = nullptr);
  [[nodiscard]] std::expected<model, result> solve(
      std::initializer_list<core::lit_t> assumptions, stats* out = nullptr);

  [[nodiscard]] std::size_t variable_count() const noexcept;

private:
  struct impl;
  std::unique_ptr<impl> p_;
};

}  // namespace satx::solver

