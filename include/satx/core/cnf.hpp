#pragma once
// satx::core — arena CNF: almacén de cláusulas normalizadas y deduplicadas

#include "clause.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace satx::core {

namespace detail {

struct lit_vec_hash {
  std::size_t operator()(std::vector<lit_t> const& v) const noexcept {
    // FNV-1a
    std::size_t h = 1469598103934665603ULL;
    for (lit_t l : v) {
      h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(l));
      h *= 1099511628211ULL;
    }
    return h;
  }
};

}  // namespace detail

// Almacén de cláusulas. Invariantes:
//   · toda cláusula está ordenada por |lit|, sin duplicados y sin tautologías;
//   · cláusulas idénticas (hash por lits) se deduplican;
//   · una cláusula vacía o la unidad [false_lit] marcan la fórmula como UNSAT;
//   · literales inválidos (0, INT32_MIN) se rechazan al añadir (ADR-011).
class cnf {
public:
  cnf() = default;

  void add_clause(std::vector<lit_t> lits) {
    for (lit_t l : lits) {
      if (l == 0 || l == std::numeric_limits<lit_t>::min())
        throw std::invalid_argument("satx::core::cnf: invalid literal (0 / INT32_MIN)");
    }
    if (lits.empty()) {
      unsat_ = true;
      return;
    }
    normalize(lits);
    if (lits.empty()) {  // tautología eliminada
      return;
    }
    if (lits.size() == 1 && lits[0] == false_lit) {  // unidad ¬VERDADERO
      unsat_ = true;
      return;
    }
    if (seen_.insert(lits).second) {
      clauses_.push_back(clause{std::move(lits)});
    }
  }

  void reserve(std::size_t n) {
    clauses_.reserve(n);
    seen_.reserve(n * 2 + 16);
  }

  [[nodiscard]] std::size_t variable_count() const noexcept { return nvars_; }
  [[nodiscard]] std::size_t clause_count() const noexcept { return clauses_.size(); }
  [[nodiscard]] bool unsat() const noexcept { return unsat_; }

  [[nodiscard]] std::span<const clause> clauses() const noexcept { return clauses_; }

  auto begin() const noexcept { return clauses_.begin(); }
  auto end() const noexcept { return clauses_.end(); }

private:
  void normalize(std::vector<lit_t>& lits) {
    std::sort(lits.begin(), lits.end(),
              [](lit_t a, lit_t b) { return var_of(a) < var_of(b); });
    std::vector<lit_t> out;
    out.reserve(lits.size());
    for (lit_t l : lits) {
      if (!out.empty() && out.back() == l) {
        continue;  // duplicado
      }
      if (!out.empty() && out.back() == neg(l)) {
        out.clear();  // tautología: descartar la cláusula completa
        break;
      }
      out.push_back(l);
    }
    lits = std::move(out);
    for (lit_t l : lits) {
      const std::size_t v = static_cast<std::size_t>(var_of(l));
      if (v > nvars_) nvars_ = v;
    }
  }

  std::vector<clause> clauses_;
  std::unordered_set<std::vector<lit_t>, detail::lit_vec_hash> seen_;
  std::size_t nvars_ = 0;
  bool unsat_ = false;
};

}  // namespace satx::core
