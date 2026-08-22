// satx::solver — implementación del puente sobre Kerberos (kernel CDCL SLIME)

#include <satx/solver/kerberos.hpp>

#include <kerberos/slime_bridge.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace satx::solver {

namespace {

struct marshalled {
  std::vector<int> flat;
  std::vector<int> sizes;
  std::vector<const int*> ptrs;
  int nvars = 0;
};

marshalled marshal(engine const& e) {
  marshalled m{};
  m.nvars = static_cast<int>(e.variable_count());
  auto const& clauses = e.formula().clauses();
  std::size_t total_lits = 0;
  for (auto const& c : clauses) total_lits += c.lits.size();
  m.flat.reserve(total_lits);
  m.sizes.reserve(clauses.size());
  m.ptrs.reserve(clauses.size());
  for (auto const& c : clauses) {
    if (c.lits.empty()) continue;  // cnf nunca almacena cláusulas vacías (defensivo)
    m.sizes.push_back(static_cast<int>(c.lits.size()));
    const std::size_t off = m.flat.size();
    for (core::lit_t l : c.lits) m.flat.push_back(static_cast<int>(l));
    m.ptrs.push_back(m.flat.data() + off);
  }
  return m;
}

SlimeSatOptions to_c(options const& opt) {
  SlimeSatOptions o{};
  o.heuristic_mode = opt.heuristic_mode;
  o.use_mab = opt.use_mab;
  o.mabc = opt.mabc;
  o.use_hess = opt.use_hess;
  o.use_ct = opt.use_ct;
  o.ct_lbd_max = opt.ct_lbd_max;
  o.ct_maxlen = opt.ct_maxlen;
  o.ct_max_cubes = opt.ct_max_cubes;
  o.ct_buddy_merge = opt.ct_buddy_merge;
  o.ct_escape_rounds = opt.ct_escape_rounds;
  o.ct_probe_restarts = opt.ct_probe_restarts;
  o.use_simplify = opt.use_simplify;
  o.use_bve = opt.use_bve;
  o.use_chrono = opt.use_chrono;
  o.use_inprocess = opt.use_inprocess;
  o.use_probe = opt.use_probe;
  return o;
}

void from_stats(SlimeSatStats const& s, stats& out) {
  out.clauses = s.clauses;
  out.learnt = s.learnt;
  out.conflicts = s.conflicts;
  out.decisions = s.decisions;
  out.propagations = s.propagations;
  out.restarts = s.restarts;
  out.hess_calls = s.hess_calls;
  out.hess_sat_hits = s.hess_sat_hits;
  out.ct_added = s.ct_added;
  out.ct_merged = s.ct_merged;
  out.ct_escaped = s.ct_escaped;
  out.ct_probe_added = s.ct_probe_added;
}

// Valida suposiciones: literales ±v con 1 <= v <= nvars (sin 0 ni INT32_MIN).
void check_assumptions(std::span<core::lit_t const> assumptions, int nvars) {
  for (core::lit_t l : assumptions) {
    if (l == 0 || l == std::numeric_limits<core::lit_t>::min())
      throw std::invalid_argument("satx::solver: invalid assumption literal (0 / INT32_MIN)");
    const std::int64_t v = l < 0 ? -static_cast<std::int64_t>(l) : l;
    if (v < 1 || v > nvars)
      throw std::invalid_argument("satx::solver: assumption literal out of range [1, nvars]");
  }
}

}  // namespace

std::expected<model, result> solve(engine const& e, std::size_t budget) {
  (void)budget;  // advisory; el puente embebido no impone límite en esta etapa (ADR-009)
  return solve(e, options{}, nullptr);
}

std::expected<model, result> solve(engine const& e, options const& opt, stats* out) {
  return solve(e, std::span<core::lit_t const>{}, opt, out);
}

std::expected<model, result> solve(engine const& e, std::span<core::lit_t const> assumptions,
                                   options const& opt, stats* out) {
  if (e.unsat()) return std::unexpected(result::unsat);

  const marshalled m = marshal(e);
  if (m.nvars < 1) throw std::runtime_error("satx::solver: engine has no variables");
  check_assumptions(assumptions, m.nvars);

  const SlimeSatOptions copt = to_c(opt);
  SlimeSatHandle* handle = slime_sat_handle_create(
      m.nvars, static_cast<int>(m.ptrs.size()), m.ptrs.data(), m.sizes.data(), &copt);
  if (handle == nullptr) {
    throw std::runtime_error("satx::solver: kerberos (slime) failed to initialize");
  }

  std::vector<std::uint8_t> model01(static_cast<std::size_t>(m.nvars), 0);
  std::vector<int> assum;
  assum.reserve(assumptions.size());
  for (core::lit_t l : assumptions) assum.push_back(static_cast<int>(l));

  SlimeSatStats cstats{};
  const int rc = slime_sat_handle_solve(handle, assum.empty() ? nullptr : assum.data(),
                                        static_cast<int>(assum.size()), &cstats,
                                        model01.data());
  slime_sat_handle_destroy(handle);

  if (out != nullptr) from_stats(cstats, *out);

  if (rc == 10) return model(std::move(model01));
  if (rc == 20) return std::unexpected(result::unsat);
  std::fprintf(stderr, "DEBUG rc=%d nvars=%d nclauses=%d\n", rc, m.nvars,
               static_cast<int>(m.ptrs.size()));
  throw std::runtime_error("satx::solver: kerberos (slime) internal error (rc=" +
                           std::to_string(rc) + ")");
}

std::expected<model, result> solve(engine const& e,
                                   std::initializer_list<core::lit_t> assumptions,
                                   options const& opt, stats* out) {
  return solve(e, std::span<core::lit_t const>(assumptions.begin(), assumptions.size()), opt,
               out);
}

// ── sesión incremental ──────────────────────────────────────────────────────

struct session::impl {
  marshalled m;
  SlimeSatHandle* handle = nullptr;
};

session::session(engine const& e, options const& opt) : p_(std::make_unique<impl>()) {
  if (e.unsat()) throw std::runtime_error("satx::solver: formula is UNSAT (empty clause)");
  p_->m = marshal(e);
  if (p_->m.nvars < 1) throw std::runtime_error("satx::solver: engine has no variables");
  const SlimeSatOptions copt = to_c(opt);
  p_->handle = slime_sat_handle_create(p_->m.nvars, static_cast<int>(p_->m.ptrs.size()),
                                       p_->m.ptrs.data(), p_->m.sizes.data(), &copt);
  if (p_->handle == nullptr) {
    throw std::runtime_error("satx::solver: kerberos (slime) failed to initialize");
  }
}

session::session(session&&) noexcept = default;
session& session::operator=(session&&) noexcept = default;

session::~session() {
  if (p_ && p_->handle != nullptr) {
    slime_sat_handle_destroy(p_->handle);
    p_->handle = nullptr;
  }
}

std::expected<model, result> session::solve(std::span<core::lit_t const> assumptions,
                                            stats* out) {
  check_assumptions(assumptions, p_->m.nvars);
  std::vector<std::uint8_t> model01(static_cast<std::size_t>(p_->m.nvars), 0);
  std::vector<int> assum;
  assum.reserve(assumptions.size());
  for (core::lit_t l : assumptions) assum.push_back(static_cast<int>(l));

  SlimeSatStats cstats{};
  const int rc = slime_sat_handle_solve(p_->handle, assum.empty() ? nullptr : assum.data(),
                                        static_cast<int>(assum.size()), &cstats,
                                        model01.data());
  if (out != nullptr) from_stats(cstats, *out);

  if (rc == 10) return model(std::move(model01));
  if (rc == 20) return std::unexpected(result::unsat);
  throw std::runtime_error("satx::solver: kerberos (slime) internal error (rc=" +
                           std::to_string(rc) + ")");
}

std::expected<model, result> session::solve(std::initializer_list<core::lit_t> assumptions,
                                            stats* out) {
  return solve(std::span<core::lit_t const>(assumptions.begin(), assumptions.size()), out);
}

std::size_t session::variable_count() const noexcept {
  return p_ ? static_cast<std::size_t>(p_->m.nvars) : 0;
}

}  // namespace satx::solver

