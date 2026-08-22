#pragma once
// satx::engine — asignador de variables + base de cláusulas (sin estado global)

#include "cnf.hpp"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <span>

namespace satx {

// Motor de construcción de circuitos. Es un objeto explícito que se pasa por
// referencia; NO resuelve: la resolución es responsabilidad de satx::solver
// (puente sobre el kernel CDCL SLIME de Kerberos).
//
// ADR-011: const_policy fue eliminado — la codificación de constantes es
// siempre estricta (std::out_of_range fuera de la caja NB) y el pliegue wrap
// se obtiene con complex::from_raw_wrap. width_policy rige la alineación de
// anchos en los operadores de bit-vector (sign_extend por defecto, el
// comportamiento requerido por CBE).
class engine {
public:
  enum class width_policy { sign_extend, truncate };

  engine() {
    // ADR-004/ADR-007: la variable 1 es la constante VERDADERO.
    (void)add_variable();
    add_clause(std::span<const core::lit_t>(&core::true_lit, 1));
  }

  engine(engine const&) = delete;
  engine& operator=(engine const&) = delete;
  engine(engine&&) = delete;
  engine& operator=(engine&&) = delete;

  // Nueva variable booleana (devuelve su literal positivo).
  [[nodiscard]] core::lit_t add_variable() {
    ++nvars_;
    return static_cast<core::lit_t>(nvars_);
  }

  void add_clause(std::span<const core::lit_t> lits) {
    formula_.add_clause(std::vector<core::lit_t>(lits.begin(), lits.end()));
    nvars_ = std::max(nvars_, formula_.variable_count());
  }

  void add_clause(std::initializer_list<core::lit_t> lits) {
    add_clause(std::span<const core::lit_t>(lits.begin(), lits.size()));
  }

  void add_unit(core::lit_t l) {
    add_clause(std::span<const core::lit_t>(&l, 1));
  }

  [[nodiscard]] std::size_t variable_count() const noexcept { return nvars_; }
  [[nodiscard]] std::size_t clause_count() const noexcept { return formula_.clause_count(); }
  [[nodiscard]] bool unsat() const noexcept { return formula_.unsat(); }

  [[nodiscard]] core::cnf const& formula() const noexcept { return formula_; }

  void set_width_policy(width_policy p) noexcept { width_policy_ = p; }
  [[nodiscard]] width_policy get_width_policy() const noexcept { return width_policy_; }

private:
  core::cnf formula_;
  std::size_t nvars_ = 0;
  width_policy width_policy_ = width_policy::sign_extend;
};

}  // namespace satx
