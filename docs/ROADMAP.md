# SATX Roadmap (v0.4.0+)

SATX is a **CNF compiler + SAT-oriented constraint toolkit** embedded in Python, aimed at **proof-oriented validation** of discrete models: if you can state it as constraints, SATX can compile it to CNF and let a solver decide satisfiable/unsatisfiable.

This roadmap is intentionally **high-level**: it describes *what* will be built and *why*, without disclosing implementation details that are part of the project’s competitive edge.

---

## 0) What “verification” means here (and what it does *not* mean)

### Definitions
- **Model**: a set of constraints (CNF clauses) generated from a specification.
- **SAT**: there exists an assignment that satisfies all constraints (a *witness* / model).
- **UNSAT**: no assignment can satisfy all constraints (a *refutation*).

### Certificates
- **SAT certificate**: the satisfying assignment (plus a **re-check** that it satisfies the CNF).
- **UNSAT certificate**: a solver-produced proof (e.g., DRAT/FRAT or equivalent), plus a **proof checker**.

### Critical boundary
SATX verifies **properties of the formal model** you wrote (or generated). It does **not** magically verify reality; it verifies the logical consequences of explicit assumptions.

---

## 1) Project vision

**Constraint-first engineering**:
- encode invariants once,
- compile to CNF deterministically,
- solve with standard solvers,
- attach verifiable certificates,
- treat results as “proof-carrying artifacts” you can ship, audit, and diff.

SATX wants to be the layer that turns “this should be true” into “this is true *given these assumptions*”.

---

## 2) Non-goals (explicit)

- **Not** a symbolic theorem prover.
- **Not** a “general AI reasoner”.
- **Not** a guarantee about real-world truth without explicit modeling assumptions.
- **Not** a replacement for legal advice (contract analysis is a *formal consistency* tool, not a court oracle).

---

## 3) Roadmap by milestones

### v0.4.x — Stabilization & consistency (current line)
Focus: correctness, determinism, developer ergonomics.
- Version unification (single source of truth: code/package/docs).
- Repo-wide license consistency checks and automation.
- Deterministic CNF output options (stable variable naming / stable clause ordering where feasible).
- Expanded regression suite (unit tests + golden CNF snapshots).
- Better error surfaces for modeling mistakes (types, scaling, overflow domains).
- “Small but sharp” documentation: minimal examples that cover 80% of usage.

Deliverable theme: **“reproducible modeling + stable artifacts”**.

---

### v0.5 — Proof-carrying SATX
Focus: turning SAT/UNSAT into auditable artifacts.
- SAT witness re-checker (verify a model assignment satisfies CNF).
- UNSAT proof pipeline: integrate solvers that emit proofs + include a proof checker workflow.
- Standard artifact bundle:
  - model spec hash
  - CNF hash
  - solver metadata
  - certificate (SAT assignment or UNSAT proof)
  - verification report
- Unsat-core / assumption-based workflows (where supported by solver backend).

Deliverable theme: **“don’t trust—verify”**.

---

### v0.6 — Verified optimization & explanations
Focus: optimization and human-readable traces without weakening rigor.
- MaxSAT / weighted constraints workflows (where appropriate).
- Incremental solving patterns (assumptions, scenarios, counterfactuals).
- Explanation layer:
  - minimal unsat cores (where feasible),
  - “why this fails” summaries grounded in cores,
  - delta-debugging helpers to shrink failing models.

Deliverable theme: **“find the breaking point fast”**.

---

### v0.7 — Specification interfaces (including LLM-assisted modeling)
Focus: making modeling accessible **without** delegating correctness to the LLM.
- A typed mini-DSL / schema for constraints (human-friendly but formal).
- Optional “assistant” interface that:
  - proposes constraint templates,
  - converts structured text/specs into constraints,
  - generates test cases and adversarial scenarios,
  - **never** overrides solver results.

Safety invariant: **LLM suggests → SATX decides**.

Deliverable theme: **“LLM as a compiler front-end; SAT as the judge”**.

---

### v1.0 — SATX as a verification substrate
Focus: long-term API stability and integration into real pipelines.
- SemVer stability commitments for public API.
- Tooling integration:
  - CI hooks (fail builds on UNSAT, attach certificates),
  - artifact caching (CNF + proofs),
  - reproducible builds.
- Performance initiatives:
  - encoding-level improvements,
  - backend adapters,
  - profiling guides.

Deliverable theme: **“industrial-grade, reproducible verification”**.

---

## 4) Backends & solver policy

SATX is solver-agnostic at the interface level:
- produces standard CNF artifacts,
- integrates with external solvers via adapters.

Policy:
- Prefer backends that support **proof emission** (for UNSAT) and stable outputs.
- Provide a simple “bring your own solver” path as a first-class option.

---

## 5) Licensing & contributions (summary)

- The repository license terms are defined at the project root (LICENSE/NOTICE, plus any commercial terms in COMMERCIAL.md if present).
- Contributions must be compatible with the repo’s root license terms.
- If you need a different licensing regime for embedding SATX in closed products, use the commercial licensing path.

(See the root documents for the authoritative legal text.)

---

## 6) How to read SATX results in practice (recommended discipline)

When you publish results, include:
- the assumptions (domain bounds, scaling, signedness, overflows),
- the exact SATX version,
- the solver version,
- the certificate (assignment or proof) and verification logs.

This is the difference between “it worked on my machine” and “it is auditable”.

---

## 7) Public tracking

Issues and milestones are used for:
- roadmap items,
- bug reports,
- minimal reproducible examples,
- proof/certificate support targets.

If you want to contribute, start with:
- new regression tests,
- artifact determinism,
- certificate tooling.

---
