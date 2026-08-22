#pragma once
// satx — API pública: re-exporta los operadores y fábricas de satx::num al namespace satx

#include "num/complex.hpp"

namespace satx {

using core::lit_t;

using num::complex;

using num::operator+;
using num::operator-;
using num::operator*;
using num::operator/;
using num::operator==;
using num::operator!=;

using num::conj;
using num::neg;
using num::real;
using num::imag;
using num::eq_lit;

// Etapa 2 (§10.7)
using num::abs_sq;
using num::pow;
using num::root_cbe;
using num::lt_lit;
using num::le_lit;

}  // namespace satx
