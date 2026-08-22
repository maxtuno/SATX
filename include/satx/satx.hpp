#pragma once
// satx — fachada única de la librería (consumo: #include <satx/satx.hpp>)

#include "core/lit.hpp"
#include "core/clause.hpp"
#include "core/cnf.hpp"
#include "core/engine.hpp"

#include "gates/primitive.hpp"
#include "gates/bitvec.hpp"

#include "num/negabinary.hpp"
#include "num/fixed_point.hpp"
#include "num/complex.hpp"

#include "solver/model.hpp"
#include "solver/kerberos.hpp"

#include "quantum/quantum.hpp"

#include "api.hpp"
