# Manual de usuario de la API satx

**satx — Librería de números complejos CBE (Complejo Binario Entrelazado) resolubles por SAT**

**Copyright (c) 2026 Oscar Riveros. Todos los derechos reservados.** Licencia dual: Apache-2.0 para uso personal; portes a otros lenguajes requieren licencia comercial con autorización expresa del autor (ver LICENSE.txt).

> **Alcance de este documento.** Manual de uso de la API pública: qué incluir, qué
> funciones hay, qué devuelven y qué excepciones lanzan. La especificación normativa
> (formato CBE, circuitos, políticas de escala y redondeo, ADR) está en
> [`architecture.md`](architecture.md); este manual la cita como «la normativa».

---

## 1. Qué es satx

`satx` es una librería autocontenida de C++26 para aritmética de **números complejos en
formato CBE** — **C**omplejo **B**inario **E**ntrelazado —, el formato numérico
**original de Oscar Riveros (2026)** definido en «Modelo Unificado de Cómputo Clónico»
(no es la representación clásica de números complejos). Una palabra de `2W` bits
`Z[2W−1:0]` representa `z = a + i·b` con **entrelazado por cableado**: los bits pares
son el carril real (`Z[2k] = R[k]`) y los bits impares el carril imaginario
(`Z[2k+1] = I[k]`); cada carril codifica en **base −2 (negabinaria)** con `F`
posiciones fraccionarias y exponente de posición `e(k) = k − F`:

```
Re(Z) = Σ R[k]·(−2)^(k−F),   Im(Z) = Σ I[k]·(−2)^(k−F),   Z = Re(Z) + i·Im(Z)
```

Forma equivalente: `z = (re + i·im) · 2^(−F)`, con `re, im ∈ [min_NB(W), max_NB(W)]` y
paso numérico mínimo `2^(−F)`. La representación es **biyectiva** (toda palabra de
`2W` bits es un número válido y único: el conjunto representable contiene exactamente
`2^(2W)` puntos de un reticulado de paso `2^(−F)`), lo que permite operar de dos
maneras sobre el mismo tipo:

- **Ruta concreta** — operaciones sobre constantes, evaluadas `constexpr` en el host con
  aritmética exacta (`__int128`).
- **Ruta simbólica** — las incógnitas son variables booleanas libres; cada operación
  compila un **circuito** (gates Tseitin → CNF) que un kernel SAT resuelve. El solver es
  el **kernel CDCL SLIME de Kerberos**, integrado como parte del producto (biblioteca
  estática `satx_kerberos`), sin solvers de terceros y sin puentes a Python.

Además del núcleo numérico, satx incluye un **subsistema cuántico** (`satx::quantum`)
que reutiliza la misma aritmética CBE: amplitudes como `complex<W,F>`, compuertas como
matrices CBE, circuitos y el bucle OTOC de *Quantum Echoes*, tanto en ruta concreta
(simulación de punto fijo) como simbólica (problema inverso resuelto por SAT).

### 1.1 Requisitos

| Requisito | Detalle |
|---|---|
| Compilador | GCC o Clang con soporte C++26 (la ruta concreta usa `__int128`; MSVC no soportado) |
| CMake | ≥ 3.28 |
| C | C17 (solo para el kernel de Kerberos, que ya viene en el repositorio) |
| Dependencias | Ninguna externa: solo la biblioteca estándar y el kernel propio |

### 1.2 Cabeceras

```cpp
#include <satx/satx.hpp>      // fachada única: core, gates, num, solver, quantum, api
```

Las cabeceras específicas (`<satx/core/engine.hpp>`, `<satx/quantum/quantum.hpp>`,
etc.) también pueden incluirse individualmente, pero el caso normal es solo la fachada.
El núcleo numérico es **header-only** (objetivo CMake `satx`); solo resolver requiere
enlazar el puente `satx_solver`.

---

## 2. Compilación e integración

### 2.1 Compilar la librería y sus tests

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Objetivos generados (los ejecutables salen en `bin/`):

| Objetivo | Tipo | Descripción |
|---|---|---|
| `satx` | INTERFACE (header-only) | Núcleo C++26: core, gates, num, quantum |
| `satx_kerberos` | STATIC (C17) | Kernel y backends de Kerberos (SLIME, BASILISK, PIXIE, WMIBO, GRINDER) |
| `satx_solver` | STATIC (C++) | Puente `satx::solver::solve` → C API de SLIME |
| `kerberos` | Ejecutable | Despachador CLI de Kerberos (opción `SATX_BUILD_KERBEROS_CLI`, ON) |
| `101` | Ejecutable | Ejemplo mínimo `x + y == z` |
| `test_*` | Ejecutables | Suite sin framework (asserts + bucles de propiedad) |

Opciones de configuración: `SATX_BUILD_TESTS` (ON por defecto) y
`SATX_BUILD_KERBEROS_CLI` (ON por defecto).

### 2.2 Consumir satx desde tu proyecto

Sin necesidad de instalar, con `add_subdirectory` (o `FetchContent`):

```cmake
add_subdirectory(satx)                 # o fetch_content de tu repositorio

add_executable(mi_app mi_app.cpp)
target_link_libraries(mi_app PRIVATE satx satx_solver)   # satx_solver solo si usas solve()
```

Regla práctica: si tu programa llama a `satx::solver::solve`, enlaza **ambos** objetivos
(`satx` y `satx_solver`); si solo usa aritmética de constantes o el subsistema cuántico
concreto, basta con `satx`.

`cmake --install` instala cabeceras, las bibliotecas estáticas `satx_kerberos`/`satx_solver`
y la documentación (`docs/`), aunque sin paquete `find_package` por ahora.

---

## 3. Conceptos previos: las dos rutas y el `engine`

### 3.1 El `engine`

El `satx::engine` es el **asignador de variables booleanas y la base de cláusulas CNF**.
No resuelve: acumula la fórmula que los circuitos generan y que luego entrega al solver.

- Es un objeto **explícito, no copiable ni movible**: se pasa por referencia.
- Al construirse fija la variable 1 como la constante VERDADERO (cláusula unitaria
  `[+1]`), por lo que los literales constantes no consumen variables.
- Todo objeto simbólico recuerda su motor (`engine_of()`); una operación entre objetos
  de motores distintos lanza `std::logic_error`.
- Un problema → un `engine` → un `solve`.

### 3.2 Ruta concreta vs. ruta simbólica

```cpp
satx::complex<16, 4> a{1.5, -2.0};          // constante: plegada, constexpr
satx::complex<16, 4> b{0.5,  3.0};
constexpr auto s = a + b;                   // aritmética del host, sin circuitos

satx::engine e;
satx::complex<16, 4> x{e};                  // incógnita: 2W variables booleanas libres
auto t = (x + b) * x;                       // circuito CNF añadido al engine e
```

El tipo oculta la dualidad: `s` y `t` son el mismo tipo `complex<16,4>`. La
representación física (`concrete`, `nb` o `tc`) es inspeccionable con
`representation()`.

### 3.3 Precisión y rango

- **Resolución:** `2^(−F)` (punto fijo binario). `F` bits fraccionarios; `F <= W <= 60`.
- **Rango negabinario** de cada componente (asimétrico, §7.1 de la normativa):

  | W | min_NB(W) | max_NB(W) |
  |---|---|---|
  | 8  | −170      | 85        |
  | 12 | −2730     | 1365      |
  | 16 | −43690    | 21845     |
  | 32 | −2863311530 | 1431655765 |

  Fórmulas: `min_NB(W) = −2·(4^⌊W/2⌋ − 1)/3`, `max_NB(W) = (4^⌈W/2⌉ − 1)/3`.

- Codificar una constante fuera del rango lanza `std::out_of_range` (la codificación
  es **siempre estricta** — ADR-011); la variante `from_raw_wrap` pliega mod `2^W` al
  rango NB.
- **Nota sobre `F`:** la unidad `2^F` debe caber en el rango NB para que
  `one()`/`i_unit()` sean construibles; con `F` grande respecto a `W` estas fábricas
  lanzan `std::out_of_range` en tiempo de ejecución (p. ej. `F = W` siempre). El
  subsistema cuántico exige `F + 2 <= W` por eso (§10.3).

---

## 4. Núcleo booleano (`satx::core` + `satx::engine`)

### 4.1 Literales

```cpp
namespace satx::core {
  using lit_t = std::int32_t;                 // 0 = inválido; ±v = variable v (v ≥ 1)
  inline constexpr lit_t true_lit  =  1;      // constante VERDADERO
  inline constexpr lit_t false_lit = -1;      // constante FALSO

  constexpr std::int32_t var_of(lit_t l);     // |l|
  constexpr lit_t neg(lit_t l);               // −l
  constexpr bool sign(lit_t l);               // l > 0
  constexpr bool is_constant(lit_t l);        // l == ±1
}
```

`satx::lit_t` está re-exportado al namespace `satx`.

### 4.2 Motor

```cpp
namespace satx {
class engine {
public:
  engine();                                              // fija la variable 1 = VERDADERO

  [[nodiscard]] core::lit_t add_variable();              // nueva variable (devuelve +v)
  void add_clause(std::span<const core::lit_t> lits);    // disyunción de literales
  void add_clause(std::initializer_list<core::lit_t> lits);
  void add_unit(core::lit_t l);                          // cláusula unitaria [l]

  std::size_t variable_count() const;                    // nº de variables
  std::size_t clause_count()   const;                    // nº de cláusulas
  bool unsat() const;                                     // cláusula vacía o [¬1]
  core::cnf const& formula() const;                      // CNF de solo lectura

  enum class width_policy { sign_extend, truncate };     // default: sign_extend
  void set_width_policy(width_policy);
  width_policy get_width_policy() const;
};
}
```

Notas:

- `add_clause` normaliza (ordena por `|lit|`, elimina duplicados y tautologías) y
  deduplica cláusulas idénticas. La cláusula vacía `[]` o la unidad `[¬1]` marcan UNSAT
  (ADR-011). Literales inválidos (0, `INT32_MIN`) → `std::invalid_argument` (ADR-011).
- `unsat()` es O(1) y cortocircuita al solver; `formula()` es el acceso que el puente
  de Kerberos vuelca a la C API.
- La aritmética CBE se apoya en `width_policy::sign_extend` (no cambies esta política
  para el uso numérico normal; `truncate` recorta al ancho menor en `gates::bitvec` —
  ADR-011).
- El motor **no resuelve**; la resolución es `satx::solver::solve` (§9).

### 4.3 Ejemplo: media suma con gates

```cpp
#include <satx/satx.hpp>
#include <cassert>

int main() {
    satx::engine e;
    const satx::lit_t a = e.add_variable();          // sumando
    const satx::lit_t b = e.add_variable();
    const satx::lit_t s = satx::gates::xor2(e, a, b); // suma
    const satx::lit_t c = satx::gates::and2(e, a, b); // acarreo

    e.add_unit(a);  e.add_unit(b);                    // 1 + 1 = 2
    const auto m = satx::solver::solve(e);
    assert(m.has_value());
    assert(m->get(s) == false);                        // 1 ⊕ 1 = 0
    assert(m->get(c) == true);                         // acarreo 1
}
```

---

## 5. Gates booleanos y bit-vectors (`satx::gates`)

### 5.1 Gates primitivos (Tseitin → CNF)

Cada gate crea una variable de salida `o` y emite las cláusulas de su tabla de verdad.
Los operandos constantes (`true_lit`/`false_lit`) se pliegan **sin consumir variables**
ni cláusulas (ADR-010).

| Gate | Firma | Cláusulas |
|---|---|---|
| `or2`  | `lit_t or2(engine&, lit_t a, lit_t b)` | 4 |
| `and2` | `lit_t and2(engine&, lit_t a, lit_t b)` | 4 |
| `xor2` | `lit_t xor2(engine&, lit_t a, lit_t b)` | 4 |
| `mux2` | `lit_t mux2(engine&, lit_t s, lit_t a, lit_t b)` — `s ? b : a` | 4 |
| `fas`  | `lit_t fas(engine&, lit_t a, lit_t b, lit_t ci)` — bit de suma del sumador completo | 8 |
| `fac`  | `lit_t fac(engine&, lit_t a, lit_t b, lit_t ci)` — bit de acarreo (mayoría) | 6 |

### 5.2 Rieles y operadores de bit-vector

Un **riel** es `std::vector<core::lit_t>` en orden **LSB-first** (posición 0 = bit 0).

```cpp
namespace satx::gates {
  using rail = std::vector<core::lit_t>;

  rail rca(engine&, rail a, rail b, core::lit_t cin = core::false_lit); // suma ripple-carry
  std::pair<rail, core::lit_t> rca_carry(engine&, rail a, rail b,
                                         core::lit_t cin = core::false_lit); // suma + acarreo
  rail rcs(engine&, rail a, rail b);                    // resta TC: a + ¬b + 1
  rail pmul(engine&, rail a, rail b);                   // multiplicador por productos parciales
  rail sext(rail a, std::size_t w);                     // extensión de signo (sin circuitos)
  rail zext(rail a, std::size_t w);                     // extensión con ceros
  rail shl(rail a, std::size_t k);                      // << k con wrap (sin circuitos)
  rail shr(rail a, std::size_t k);                      // >> k lógico (sin circuitos)
  rail sra(rail a, std::size_t k);                      // >> k aritmético (sin circuitos)
  rail not_rails(rail a);                               // negación bit a bit (sin circuitos)
  rail and_rails(engine&, rail a, rail b);              // AND bit a bit
  rail or_rails(engine&, rail a, rail b);               // OR bit a bit
  rail xor_rails(engine&, rail a, rail b);              // XOR bit a bit
  core::lit_t eq(engine&, rail a, rail b);              // igualdad bit a bit → literal
  core::lit_t ule(engine&, rail a, rail b);             // ≤ sin signo
  core::lit_t sle(engine&, rail a, rail b);             // ≤ con signo
  core::lit_t ult(engine&, rail a, rail b);             // < sin signo (¬ule(b,a))
  core::lit_t slt(engine&, rail a, rail b);             // < con signo (¬sle(b,a))
  rail mux(engine&, core::lit_t s, rail a, rail b);     // s ? b[i] : a[i]
  core::lit_t reduce_or(engine&, rail a);               // OR reducido (≠ 0)
  core::lit_t reduce_and(engine&, rail a);              // AND reducido (todo 1)
  bool all_constant(rail const& a);                     // predicado de host, O(W)
}
```

Notas:

- `rca`/`rcs`/`pmul` **pliegan rieles 100 % constantes** por aritmética directa del
  host (ADR-010); el pliegue de `rca`/`rca_carry` usa `__int128` (ancho ≤ 127) y el de
  `pmul` `uint64_t` (ancho ≤ 64) — sin UB de desplazamientos (ADR-011). El acarreo
  final de `rca` se descarta (wrap); `rca_carry` lo expone.
- `pmul` devuelve los `w` bits bajos; para el producto exacto de dos valores con
  signo de `w` bits hay que `sext`ear antes a `2w` (el núcleo numérico entra sus
  operandos ya extendidos — §8.3).
- `eq` sobre rieles idénticos devuelve `true_lit` sin circuito; sobre rieles constantes
  distintos, `false_lit`.
- `shl`/`shr`/`sra`/`sext`/`zext`/`not_rails` no emiten cláusulas (re-enrutado).
- `reduce_or` es la restricción «distinto de cero» que usa la división simbólica.
- Los operadores de ancho mixto respetan `width_policy` del motor.

---

## 6. Codec negabinario y punto fijo (`satx::num`)

### 6.1 `negabinary<W>` (ruta concreta, `constexpr`)

```cpp
template<std::size_t W>            // W ∈ [2, 60]
struct satx::num::negabinary {
  std::array<std::uint8_t, W> digits{};          // LSB-first
  static constexpr std::int64_t min() noexcept;  // rango NB (tabla §3.3)
  static constexpr std::int64_t max() noexcept;
  static constexpr negabinary encode(std::int64_t n);   // fuera de rango → std::out_of_range
  constexpr std::int64_t decode() const noexcept;       // Σ d_i·(−2)^i
};
```

```cpp
constexpr auto nb = satx::num::negabinary<8>::encode(-1);   // −1 = 11₂₋ (dígitos 1,1)
static_assert(nb.decode() == -1);
```

### 6.2 `fixed_point` (escala 2^F)

```cpp
namespace satx::num::fixed_point {
  constexpr std::int64_t scale(std::size_t f);                       // 2^f
  constexpr std::int64_t to_raw(double x, std::size_t f);            // round(x·2^f)
  constexpr double from_raw(std::int64_t raw, std::size_t f);        // raw/2^f
  constexpr std::int64_t trunc_shr(std::int64_t v, std::size_t k);        // >> hacia 0
  constexpr std::int64_t trunc_shr_i128(__int128 v, std::size_t k);       // ídem, 128 bits
  constexpr std::int64_t round_scaled_div(__int128 n, __int128 den,
                                          std::size_t k);            // round(n·2^k/den)
}
```

- `to_raw` redondea **half-away-from-zero** (ADR-003) y lanza `std::out_of_range` si
  `x·2^f` no cabe en `int64` (no finito incluido) — ADR-011.
- `trunc_shr*` es el recorte simétrico (hacia cero) que aplica `mul` en la ruta concreta.
- `round_scaled_div` (división larga de 256 bits) es la primitiva de `operator/`
  concreto: `den <= 0` o `k > 120` → `std::domain_error`; `|resultado| ≥ 2^62` o
  `n·2^k` con más de 241 bits → `std::overflow_error` (ADR-011).

---

## 7. El número complejo CBE (`satx::complex<W,F>`)

```cpp
template<std::size_t W, std::size_t F>
  requires (W >= 2 && W <= 60 && F <= W)
class satx::complex;                       // también satx::num::complex
```

Estáticos: `complex<W,F>::width == W`, `complex<W,F>::fractional == F`.

### 7.1 Construcción y fábricas

| Constructor / fábrica | Efecto | Ruta |
|---|---|---|
| `complex()` | constante `(0, 0)` | concreta |
| `complex(int64_t re_raw, int64_t im_raw)` | constante **ya escalada** (raw = valor·2^F); valida rango NB | concreta |
| `complex(A re, B im)` (A, B floating) | codifica `round(x·2^F)` en NB | concreta |
| `explicit complex(engine&)` | **incógnita**: `2W` variables frescas (rieles NB) | simbólica |
| `zero(engine&)` / `one(engine&)` / `i_unit(engine&)` | constantes `0`, `1`, `i` (el argumento `engine` se ignora; es uniformidad de API) | concreta |
| `from_float(double re, double im)` | como el constructor de coma flotante | concreta |
| `from_raw(int64_t re_raw, int64_t im_raw)` | constante desde valores crudos | concreta |
| `from_raw_wrap(int64_t re_raw, int64_t im_raw)` | pliega mod 2^W al rango NB | concreta |
| `from_tc_rails(engine&, gates::rail re, gates::rail im)` | promoción de rieles TC (resultados del datapath; avanzado). SEMÝNTICA: el patrón se decodifica con `wrap_nb` (ADR-011). Exige anchos exactos (`std::invalid_argument` si no) | simbólica |
| `from_nb_rails(engine&, array<lit_t,W> re, array<lit_t,W> im)` | rieles NB directos (avanzado) | simbólica |
| `wrap_nb(int64_t v)` | pliegue mod 2^W al rango NB (representante canónico de la caja) | host |

```cpp
satx::complex<16, 4> a{1.5, -2.0};            // raw: (24, −32); valor: (1.5, −2.0)
satx::complex<16, 4> b{satx::complex<16,4>::from_raw(24, -32)};   // ≡ a
assert(a == b);                               // igualdad de valor entre constantes

satx::engine e;
satx::complex<16, 4> x{e};                    // variable libre (32 variables booleanas)
satx::complex<16, 4> y{1.0, 0.0};             // constante
auto z = x + y;                               // circuito en e
```

### 7.2 Acceso e introspección

| Miembro | Descripción |
|---|---|
| `kind representation()` | `kind::concrete` \| `kind::nb` \| `kind::tc` (formato físico del objeto) |
| `bool is_concrete()` | `representation() == kind::concrete` |
| `engine* engine_of()` | motor dueño (`nullptr` en constantes) |
| `int64_t re_raw()` / `im_raw()` | valor crudo escalado; **solo constantes** (si no, `std::logic_error`) |
| `array<lit_t,W> re_pattern()` / `im_pattern()` | patrón almacenado (dígitos NB o patrón TC), sin circuitos (ADR-011) |
| `array<lit_t,W> real_rail(engine&)` / `imag_rail(engine&)` | riel **NB** canónico (convierte TC→NB bajo demanda si hace falta) |
| `array<lit_t,W> tc_real_rail(engine&)` / `tc_imag_rail(engine&)` | patrón **TC** del datapath (convierte NB→TC bajo demanda) |
| `std::complex<double> value(model const&)` | decodificación post-solve: NB directo, o patrón TC → `wrap_nb` (ADR-011), dividido por 2^F. **Detecta pérdida de precisión**: si el valor exacto raw/2^F no es representable en double (bits significativos > 53), lanza `std::overflow_error` en lugar de devolver un valor redondeado silenciosamente |
| `std::complex<int64_t> value_raw(model const&)` | valor **exacto** post-solve (componentes enteras del representante canónico NB), sin pasar por double |

```cpp
satx::engine e;
satx::complex<6, 2> x{e};
e.add_unit(satx::eq_lit(e, x, satx::complex<6, 2>{-1.25, 0.75}));  // raws (−5, 3)
if (auto m = satx::solver::solve(e); m) {
    std::cout << x.value(*m) << '\n';          // (-1.25, 0.75)
}
```

`value(model)` funciona también sobre constantes (ignora el modelo, salvo que se le pase
el modelo vacío `satx::solver::model{}`).

### 7.3 Exactitud de la conversión TC→NB

```cpp
satx::num::nb_exact_lit(engine& e, std::span<const core::lit_t> tc_rail) → lit_t;
```

Devuelve el literal «el valor con signo del riel TC está en la caja NB»
(`min_NB(W) ≤ v ≤ max_NB(W)`, comparadores en W+1 bits — ADR-006 revisado; la
condición anterior `c_final == t[W−1]` era incorrecta). Útil para detectar desbordes
NB en resultados del datapath. Nota: la recurrencia TC→NB con acarreo de un bit
anterior también producía dígitos erróneos; la conversión correcta usa el resto
completo (normativa §7.3).

---

## 8. Operaciones

Todas las operaciones son plantillas heterogéneas: operandos `complex<Wa,Fa>` y
`complex<Wb,Fb>` promueven a `W = max(Wa,Wb)`, `F = max(Fa,Fb)` **con alineación de
escala** (ADR-008: el operando de escala menor se desplaza `F−Fa` bits). Con `F` iguales
todo degenera a las fórmulas clásicas.

Regla general de ruta: si **ambos operandos son constantes** → aritmética `constexpr`
del host y re-codificación NB; si alguno es simbólico → circuito. Operar símbolos de
motores distintos (o sin motor) lanza `std::logic_error`.

### 8.1 Suma y resta

```cpp
auto s = a + b;      // re = re_a + re_b (mod 2^W, wrap), ídem im; circuito: rca
auto d = a - b;      // circuito: rcs (resta en complemento a dos)
```

**Desborde:** envolvente (wrap mod 2^W) en add/sub, **idéntico en ambas rutas**
(ADR-011: la ruta concreta pliega con `wrap_nb`, ya no lanza). El resultado simbólico
se almacena como patrón TC y `value(model)` lo decodifica al representante canónico de
la caja NB.

### 8.2 Unarias

```cpp
auto c = satx::conj(z);    // (re, −im)   — 1× RCS en ruta simbólica
auto n = satx::neg(z);     // (−re, −im)  — 2× RCS
auto r = satx::real(z);    // (re, 0)     — extracción de la parte real
auto i = satx::imag(z);    // (im, 0)     — extracción de la parte imaginaria
```

`real`/`imag` funcionan en ambas rutas; en ruta simbólica devuelven un complejo cuya
otra componente es el riel cero (`false_lit`) — útil como operandos de comparaciones
por componente (véase `examples/mandelbrot_escape.cpp`).

### 8.3 Multiplicación

```cpp
auto p = a * b;      // P_re = ar·br − ai·bi;  P_im = ar·bi + ai·br;  recorte >> K
```

Con `K = min(Fa,Fb)` (con `F` iguales, `K = F`). El recorte trunca **hacia cero**
(ADR-003/ADR-012) en **ambas** rutas: la concreta con `trunc_shr_i128`, la simbólica
con la corrección `P + (signo ? 2^K−1 : 0)` antes de recortar la ventana.

**Chequeo de desborde (ADR-011):** el cociente truncado debe caber en la **caja NB**
`[min_NB(W), max_NB(W)]`. En ruta simbólica añade restricciones de comparación con
signo (la fórmula resulta **UNSAT** si el producto no cabe); en ruta concreta lanza
`std::out_of_range` si el resultado no cabe en NB. Los operandos entran exactos al
circuito (TC con signo en W+1 bits), de modo que valores de la caja NB que exceden el
rango con signo de W bits no se corrompen.

```cpp
satx::engine e;
const satx::complex<6, 0> x{e};
e.add_unit(satx::eq_lit(e, x, satx::complex<6, 0>{10, 0}));
(void)(x * x);                                    // añade las restricciones de no-desborde
auto m = satx::solver::solve(e);
assert(!m.has_value() && m.error() == satx::solver::result::unsat);
```

### 8.4 División

Ruta concreta (ambos constantes): fórmula exacta

```
(a + bi) / (c + di) = ((ac + bd) + i(bc − ad)) / (c² + d²)
```

escalada a `2^F` con `round_scaled_div`; **división por cero → `std::domain_error`**.

Ruta simbólica: crea un complejo libre `r` e impone `mul(r, b) == a` (igualdad bit a bit
sobre la representación exacta alineada, ADR-013) **más la restricción de divisor no
nulo** sobre los rieles nativos de `b` (`b != 0`; ADR-011: con rieles re-escalados,
divisores válidos producían UNSAT espurio). Dividir por cero simbólico produce UNSAT,
no una excepción. **La división simbólica es exacta-en-cuadrícula**: cocientes no
representables en la grilla (p. ej. `1/3` en la escala 2^F) son UNSAT, no se redondean;
la Etapa 3 añadirá el divisor iterativo con redondeo:

```cpp
satx::engine e;
const satx::complex<6, 2> x{e};
e.add_unit(satx::eq_lit(e, x, satx::complex<6, 2>{-1.0, 2.0}));
const auto q = x / satx::complex<6, 2>::i_unit(e);      // q·i == x
const auto back = q * satx::complex<6, 2>::i_unit(e);
auto m = satx::solver::solve(e);
assert(m.has_value());
// back.value(*m) ≈ x.value(*m)  dentro de 2^−F
```

### 8.5 Igualdad

| Operador | Qué hace | Cuándo |
|---|---|---|
| `operator==` / `operator!=` | comparación de **valor** (alinea escalas con `__int128`) | **solo constantes**; sobre símbolos lanza `std::logic_error` |
| `eq_lit(engine&, a, b) → lit_t` | igualdad bit a bit de los rieles sobre la representación **exacta** al ancho `max(Wa+F−Fa, Wb+F−Fb)+1` (ADR-013); devuelve un **literal** | ambos operandos, concreto y simbólico |

La igualdad sobre circuitos **no** devuelve `bool` sino un literal; la impones con
`engine::add_unit`:

```cpp
e.add_unit(satx::eq_lit(e, x + y, z));       // obliga a que x + y == z
e.add_unit(-satx::eq_lit(e, a, b));          // obliga a que a != b
```

### 8.6 Operaciones derivadas (Etapa 2)

| Operación | Descripción | Coste |
|---|---|---|
| `satx::abs_sq(z) → complex<W,F>` | `z · conj(z)`; la parte imaginaria se anula exactamente (`im == 0`) | 1× mul, O(W²) |
| `satx::lt_lit(engine&, a, b) → lit_t` | comparación **lexicográfica** `(re, im)`: `re_a < re_b ∨ (re_a == re_b ∧ im_a < im_b)` | O(W) |
| `satx::le_lit(engine&, a, b) → lit_t` | ídem con `≤` | O(W) |
| `satx::pow(z, int n) → complex<W,F>` | square & multiply; `n < 0` usa `(1/z)^(−n)` (la división impone divisor no nulo); `n == 0` → `one` | O(log n)× mul |
| `satx::root_cbe(z, int n) → complex<W,F>` | `n < 1` → `std::domain_error`; concreto: raíz del host re-codificada; simbólico: `y` libre con `y^n == z` | (n−1)× mul + EQ |

- `gt`/`ge` se obtienen invirtiendo operandos: `lt_lit(e, b, a)` / `le_lit(e, b, a)`.
- `lt_lit`/`le_lit` **pliegan** a `true_lit`/`false_lit` cuando ambos operandos son
  constantes; en ruta simbólica devuelven un literal que se impone con `add_unit`.
  El mismo objeto consigo mismo pliega: `lt_lit(e, z, z) ≡ false_lit` y
  `le_lit(e, z, z) ≡ true_lit` (ADR-011). La alineación de escala de las comparaciones
  es exacta (ADR-013), igual que en `eq_lit`.
- `root_cbe` sobre un `z` concreto devuelve la constante `z^(1/n)`; para *buscar* una
  raíz con el solver construye la incógnita y la restricción a mano (o usa `root_cbe`
  sobre un `z` simbólico):

```cpp
satx::engine e;
satx::complex<6, 2> y{e};                                  // incógnita
e.add_unit(satx::eq_lit(e, y * y, satx::complex<6, 2>{4.0, 0.0}));
if (auto m = satx::solver::solve(e); m)
    std::cout << y.value(*m) << '\n';                      // p. ej. (2, 0) o (−2, 0)
```

### 8.7 Coste de circuitos (resumen)

| Operación | Celdas principales | Coste |
|---|---|---|
| add/sub | 2× RCA de W bits (+ NB→TC O(W) por operando nb) | O(W) |
| mul | 4× PM de (2W+2)×(2W+2) + 2 combinaciones + recorte y caja | O(W²) |
| div (simbólica) | 1× mul + 2× EQ exactas + divisor ≠ 0 | O(W²) |
| eq_lit / lt_lit / le_lit | EQ/SLE/AND/OR por riel (exactos, ADR-013) | O(W) |

---

## 9. El solver (`satx::solver`)

```cpp
namespace satx::solver {
  enum class result { sat, unsat };

  struct options {                  // espejo de SlimeSatOptions (defaults de slime.c)
    int heuristic_mode = 0;         // 0 = VSIDS, 1 = CHB
    int use_mab = 0;  double mabc = 4.0;
    int use_hess = 0;
    int use_ct = 1;
    int ct_lbd_max = 6;  int ct_maxlen = 12;  int ct_max_cubes = 40000;
    int ct_buddy_merge = 0;  int ct_escape_rounds = 4;  int ct_probe_restarts = 4;
  };
  struct stats {                    // espejo de SlimeSatStats
    long long clauses = 0, learnt = 0;        // instantáneas absolutas
    long long conflicts = 0, decisions = 0, propagations = 0, restarts = 0;
    long long hess_calls = 0, hess_sat_hits = 0, ct_added = 0, ct_merged = 0,
              ct_escaped = 0, ct_probe_added = 0;   // deltas por llamada
  };

  class model {
  public:
    bool get_var(std::int32_t var) const noexcept;   // acceso 1-indexado por variable
    bool get(core::lit_t l) const noexcept;          // respeta el signo del literal
    std::size_t variable_count() const noexcept;
  };

  std::expected<model, result> solve(engine const& e, std::size_t budget = 0);
  std::expected<model, result> solve(engine const& e, options const& opt,
                                     stats* out = nullptr);
  std::expected<model, result> solve(engine const& e,
                                     std::span<core::lit_t const> assumptions,
                                     options const& opt = {}, stats* out = nullptr);
  std::expected<model, result> solve(engine const& e,
                                     std::initializer_list<core::lit_t> assumptions,
                                     options const& opt = {}, stats* out = nullptr);

  class session {                    // handle CDCL reutilizado (ADR-011)
  public:
    explicit session(engine const& e, options const& opt = {});
    session(session&&) noexcept;     // movible, no copiable
    ~session();
    std::expected<model, result> solve(std::span<core::lit_t const> assumptions = {},
                                       stats* out = nullptr);
    std::expected<model, result> solve(std::initializer_list<core::lit_t> assumptions,
                                       stats* out = nullptr);
    std::size_t variable_count() const noexcept;
  };
}
```

- `solve` vuelca la CNF del `engine` a la C API embebida de SLIME (sin ficheros
  intermedios): devuelve `model` si SAT, `std::unexpected(result::unsat)` si UNSAT, y
  lanza `std::runtime_error` ante un error interno del kernel.
- **Opciones (ADR-011):** `options` es el espejo de `SlimeSatOptions` (los defaults
  coinciden con `slime_sat_options_default` de `slime.c`); permite p. ej. cambiar la
  heurística a CHB (`heuristic_mode = 1`) o activar HESS. **Estadísticas:** `stats`
  rellena los contadores del kernel; `clauses`/`learnt` son instantáneas absolutas y el
  resto deltas por llamada.
- **Suposiciones (ADR-011):** `solve(e, {¬a, b})` resuelve bajo suposiciones DIMACS
  (se revierten tras la llamada). Literales inválidos (0, `INT32_MIN`, fuera de
  `[1, nvars]`) → `std::invalid_argument`.
- **Sesiones (ADR-011):** `session` crea el handle CDCL una vez y lo reutiliza entre
  llamadas (conserva las cláusulas aprendidas); útil para consultas incrementales y
  para medir el efecto del aprendizaje. RAII destruye el handle.
- `engine::unsat()` (cláusula vacía o `[¬1]`) cortocircuita la llamada al kernel.
- `budget` es **advisory** (ADR-009): reservado; hoy el puente no impone límite de
  conflictos.
- Las variables no asignadas en el modelo se leen como `false`; los literales
  inválidos devuelven `false` en `model::get` (ADR-011).
- Como `lit_t ≡ int32_t`, el acceso por variable usa `get_var` (no hay sobrecarga con
  `get` posible en C++).
- **Conteo de modelos:** enumera con cláusulas de bloqueo (cláusula `¬(modelo)` añadida
  al `engine` tras cada solución) — véase `examples/model_counting.cpp` y
  `test_solver_api.cpp`. El backend BASILISK del CLI de Kerberos hace el conteo exacto
  sin enumerar (Etapa 5 lo conectará al puente).

Patrón de uso estándar:

```cpp
if (auto m = satx::solver::solve(e); m) {
    std::cout << x.value(*m) << '\n';           // el modelo SAT
} else {
    assert(m.error() == satx::solver::result::unsat);
}
```

---

## 10. Subsistema cuántico (`satx::quantum`)

Amplitudes, compuertas y circuitos sobre la misma aritmética CBE. Una compuerta es una
matriz de coeficientes `complex<W,F>`: `qgate2` (matriz 2×2 row-major) y `qgate4`
(matriz 4×4 row-major, índice `2·bit(q1) + bit(q2)`).

### 10.1 Tipos y fábricas básicas

```cpp
namespace satx::quantum {
  template<std::size_t W, std::size_t F> using cx = satx::num::complex<W, F>;

  template<std::size_t W, std::size_t F> cx<W, F> mk(double re, double im);  // coef. desde float

  template<std::size_t W, std::size_t F> struct qgate2 {  // 2×2 row-major
    std::array<cx<W, F>, 4> m{};
    constexpr qgate2 adjoint() const;                     // U†
  };
  template<std::size_t W, std::size_t F> struct qgate4 {  // 4×4 row-major
    std::array<cx<W, F>, 16> m{};
    constexpr qgate4 adjoint() const;
  };
}
```

### 10.2 Catálogo de compuertas

| 1 qubit | Matriz / significado |
|---|---|
| `id2<W,F>()` | identidad |
| `x2<W,F>()`, `y2<W,F>()`, `z2<W,F>()` | Pauli X, Y, Z |
| `h2<W,F>()` | Hadamard (coeficientes 1/√2) |
| `s2<W,F>()`, `t2<W,F>()` | fase S; T = diag(1, e^{iπ/4}) |
| `rx2<W,F>(theta)`, `ry2<W,F>(theta)` | rotaciones con ángulo `double` del host |

| 2 qubits | Matriz / significado |
|---|---|
| `id4<W,F>()` | identidad |
| `cz4<W,F>()` | fase −1 en `|11⟩` |
| `cnot4<W,F>()` | **q1 = control**, q2 = objetivo (intercambia `|10⟩ ↔ |11⟩`) |
| `iswap4<W,F>()` | `|01⟩ ↔ i·|10⟩` |
| `fsim4<W,F>(theta, phi)` | sub-bloque `|01⟩/|10⟩` de ángulo θ y fase condicional e^{−iφ} |

Compuertas **libres** (coeficientes simbólicos, para aprendizaje / problema inverso):

```cpp
satx::quantum::free_gate2<W, F>(engine& e) → qgate2<W, F>;   // 4 coeficientes libres
satx::quantum::free_gate4<W, F>(engine& e) → qgate4<W, F>;   // 16 coeficientes libres
```

### 10.3 Estado (`qstate<W,F>`)

Requiere `W >= 4` y `F + 2 <= W` (la unidad `2^F` debe caber en el rango NB;
`static_assert`).

```cpp
template<std::size_t W, std::size_t F>
class satx::quantum::qstate {
public:
  explicit qstate(std::size_t n);          // concreta: |0…0⟩ (amplitudes constantes)
  explicit qstate(satx::engine& e, std::size_t n);   // simbólica: amplitudes libres,
                                                     // estado inicial |0…0⟩ impuesto
  std::size_t num_qubits() const;          // n
  std::size_t size() const;                // 2^n amplitudes
  cx<W, F> const& amplitude(std::size_t x) const;
  bool is_symbolic() const;
  satx::engine* engine() const;            // nullptr en ruta concreta

  void apply1(std::size_t q, qgate2<W, F> const& g);            // qubit fuera de rango → std::out_of_range
  void apply2(std::size_t q1, std::size_t q2, qgate4<W, F> const& g);  // q1 = fila, q2 = columna
  cx<W, F> expect_z(std::size_t q) const;   // ⟨Z_q⟩ = Σ_x s_x·|ψ_x|² (im ≡ 0)
  double norm_sq_concrete() const;          // Σ|ψ_x|² — solo ruta concreta (si no, logic_error)
  std::vector<std::complex<double>> decode(satx::solver::model const& m) const;
};
```

- El constructor simbólico añade las restricciones de igualdad `amps[x] == (x==0 ? 1 : 0)`.
- `n == 0` o `n > 20` lanza `std::invalid_argument` en ambos constructores (ADR-011:
  el límite evita `1<<n` con UB y asignaciones descomunales).
- `amplitude(x)` con `x >= size()` lanza `std::out_of_range` (ADR-011).
- En ruta simbólica, `apply1`/`apply2` reemplazan cada amplitud por una expresión
  CBE (circuitos de `mul`/`add`), que luego se decodifica con `decode(model)`.

Ejemplo — estado de Bell (ruta concreta):

```cpp
#include <satx/satx.hpp>
#include <iostream>

int main() {
    using namespace satx::quantum;
    constexpr std::size_t W = 24, F = 12;
    const satx::solver::model m{};

    qstate<W, F> s{2};                  // |00⟩
    s.apply1(0, h2<W, F>());            // (|0⟩+|1⟩)|0⟩ / √2
    s.apply2(0, 1, cnot4<W, F>());      // (|00⟩+|11⟩) / √2

    const auto a = s.decode(m);
    std::cout << "|00⟩: " << a[0] << "   |11⟩: " << a[3] << '\n';
    std::cout << "Σ|ψ|² = " << s.norm_sq_concrete() << '\n';   // ≈ 1
}
```

### 10.4 Circuitos y Quantum Echoes

```cpp
template<std::size_t W, std::size_t F>
class satx::quantum::qcircuit {
public:
  void push(std::size_t q, qgate2<W, F> const& g);             // paso de 1 qubit
  void push(std::size_t q1, std::size_t q2, qgate4<W, F> const& g);
  std::size_t size() const;   bool empty() const;
  void apply_to(qstate<W, F>& s) const;          // U (orden forward)
  void apply_adjoint_to(qstate<W, F>& s) const;  // U† (orden inverso)
};

template<std::size_t W, std::size_t F>
qcircuit<W, F> random_circuit(std::size_t n, std::size_t layers, std::uint32_t seed);
// n == 0 → std::invalid_argument; usa grilla 2D si n es cuadrado perfecto, si no cadena

template<std::size_t W, std::size_t F>
cx<W, F> forward_signal(qstate<W, F> psi, qcircuit<W, F> const& U, std::size_t qM);
// C(1): U y ⟨Z⟩ en qM (psi se pasa por valor)

template<std::size_t W, std::size_t F>
cx<W, F> otoc_echo(qstate<W, F> psi, qcircuit<W, F> const& U,
                   std::size_t qB, qgate2<W, F> const& B,
                   std::size_t qM, qgate2<W, F> const& M, std::size_t k);
// k pasadas de (U → B → U† → M); k=1 → C(2), k=2 → C(4)

template<std::size_t W, std::size_t F>
void constrain_unitary2(satx::engine& e, qgate2<W, F> const& g);
// impone U†U = I (columnas ortonormales) sobre una compuerta libre (aprendizaje)
```

Ejemplo — problema inverso (aprender `B = X` desde un dato OTOC, ruta simbólica):

```cpp
#include <satx/satx.hpp>
#include <cassert>
#include <iostream>

int main() {
    using namespace satx::quantum;
    constexpr std::size_t W = 8, F = 4;

    qcircuit<W, F> U;
    U.push(0, rx2<W, F>(0.7));

    // 1. Dato observado: se mide con la MISMA aritmética simbólica (B = X fijo),
    //    para que la restricción sea consistente con la grilla de punto fijo.
    satx::engine e0;
    qstate<W, F> ref{e0, 1};
    const auto c2_true = otoc_echo<W, F>(ref, U, 0, x2<W, F>(), 0, id2<W, F>(), 1);
    const auto s0 = satx::solver::solve(e0);
    assert(s0.has_value());
    const std::complex<double> target = c2_true.value(*s0);
    const auto target_c = cx<W, F>{target.real(), target.imag()};

    // 2. Problema inverso: B libre + unitariedad, OTOC == dato.
    satx::engine e;
    qstate<W, F> psi{e, 1};                    // |0⟩ simbólico
    const auto B = free_gate2<W, F>(e);        // compuerta desconocida
    for (std::size_t i = 0; i < 4; ++i)
        for (auto l : B.m[i].imag_rail(e)) e.add_unit(-l);   // B con coeficientes reales
    constrain_unitary2(e, B);                  // B ∈ O(2)

    const auto c2 = otoc_echo<W, F>(psi, U, 0, B, 0, id2<W, F>(), 1);
    e.add_unit(satx::eq_lit(e, c2, target_c)); // dato observado

    if (auto m = satx::solver::solve(e); m) {
        for (auto const& b : B.m)
            std::cout << b.value(*m) << '\n';  // coeficientes aprendidos de B
    }
}
```

> Nota (ADR-011): fijar el dato a un valor arbitrario (p. ej. `-0.5` en la grilla
> F=4) puede ser UNSAT si ese valor no es alcanzable por la aritmética de punto fijo;
> medir el dato con el mismo instrumento (como arriba) garantiza consistencia.

---

## 11. Errores y excepciones

| Excepción | Cuándo |
|---|---|
| `std::out_of_range` | Constante fuera del rango NB al construir (codificación estricta, ADR-011) o fuera de `int64` en `to_raw`; resultado concreto de `mul` o `root_cbe` que no cabe en NB; `one()`/`i_unit()` con `F` tal que `2^F` no cabe en NB; qubit fuera de rango en `apply1`/`apply2`/`expect_z`; `amplitude(x)` fuera de rango; `nb_exact_lit`/`tc_to_signed` con ancho > 60 |
| `std::logic_error` | `re_raw()`/`im_raw()` sobre un complejo simbólico; `operator==`/`!=` sobre símbolos (usa `eq_lit`); operación simbólica sin `engine` u operandos de motores distintos; `norm_sq_concrete()` sobre estado simbólico |
| `std::domain_error` | División concreta por cero; `root_cbe` con `n < 1`; `round_scaled_div` con `den <= 0` o `k > 120` |
| `std::invalid_argument` | Literales inválidos (0, `INT32_MIN`) en `add_clause`/suposiciones (ADR-011); suposiciones fuera de `[1, nvars]`; `qstate` con `n == 0` o `n > 20`; `random_circuit` con `n == 0`; `from_tc_rails` con rieles de ancho incorrecto |
| `std::overflow_error` | `round_scaled_div` con resultado ≥ 2^62 o `n·2^k` de más de 241 bits |
| `std::runtime_error` | Error interno del kernel SLIME (retorno inesperado) |

**UNSAT no es una excepción:** una fórmula insatisfacible produce
`std::unexpected(result::unsat)` en `solve` (o `engine::unsat() == true` si hay cláusula
vacía o `[¬1]`). Multiplicaciones que desbordan la caja NB, divisiones por cero
simbólicas y cocientes no representables en la grilla se manifiestan como UNSAT, no
como excepciones.

---

## 12. Recetas completas

> **Ejemplos autocontenidos:** la carpeta `examples/` del repositorio contiene
> programas completos y comentados, compilados junto al proyecto:
> clásicos (`send_more_money`, `map_coloring`, `nqueens`, `optimize_sum`,
> `job_shop`, `dice_distribution`, `projectile`), cuánticos (`quantum_bell`,
> `quantum_learning`, `quantum_teleportation`) y **avanzados** (revisión ADR-011):
> `gaussian_integer_factorization` (factorización en Z[i] por SAT con enumeración),
> `mandelbrot_escape` (búsqueda de parámetros con tiempo de escape exacto, heurística
> CHB vía `solver::options`), `sudoku` (9×9 con alldifferent por `¬eq_lit`),
> `model_counting` (conteo de modelos por cláusulas de bloqueo + sesiones
> incrementales) y `complex_polynomial_roots` (raíces de un polinomio CBE por SAT).

### 12.1 «Hola mundo» SAT (equivalente a `101.cpp`)

```cpp
#include <satx/satx.hpp>
#include <iostream>

int main() {
    using C = satx::complex<16, 4>;

    satx::engine e;
    const C x{e}, y{e}, z{e};                       // tres incógnitas

    e.add_unit(satx::eq_lit(e, x + y, z));          // restricción: x + y == z

    if (auto s = satx::solver::solve(e); s) {
        std::cout << "x = " << x.value(*s) << '\n'
                  << "y = " << y.value(*s) << '\n'
                  << "z = " << z.value(*s) << '\n'
                  << "x + y = " << x.value(*s) + y.value(*s) << '\n';
    }
}
```

### 12.2 Identidades en ruta concreta (`constexpr`)

```cpp
#include <satx/satx.hpp>
#include <cmath>

constexpr bool identidades() {
    using C = satx::complex<16, 8>;
    const C i = C::from_raw(0, 256);                 // (0 + 1i)·2^8, es decir, i
    return i * i == C{-1.0, 0.0}                     // i·i == −1
        && satx::abs_sq(C{3.0, 4.0}) == C{25.0, 0.0};
}
static_assert(identidades());
```

> Nota: las fábricas `zero/one/i_unit` reciben `engine&`; en un contexto `constexpr`
> sin motor conviene `from_raw` (o el constructor `complex(re_raw, im_raw)`), que no
> exigen `engine`. El parámetro de las fábricas solo importa en contextos simbólicos.

### 12.3 Buscar un complejo que cumpla una propiedad

```cpp
satx::engine e;
const satx::complex<8, 2> x{e};
e.add_unit(satx::eq_lit(e, x * x + x, satx::complex<8, 2>{2.0, 0.0}));  // x² + x == 2
if (auto m = satx::solver::solve(e); m)
    std::cout << "raíz: " << x.value(*m) << '\n';        // p. ej. (1, 0) o (−2, 0)
```

### 12.4 Comparación lexicográfica

```cpp
satx::engine e;
const satx::complex<6, 2> x{e}, y{e};
e.add_unit(satx::eq_lit(e, x, satx::complex<6, 2>{1.0, 2.0}));
e.add_unit(satx::eq_lit(e, y, satx::complex<6, 2>{1.0, 3.0}));
e.add_unit(satx::lt_lit(e, x, y));                     // (1,2) < (1,3)
assert(satx::solver::solve(e).has_value());
```

### 12.5 Solver avanzado: opciones, suposiciones, sesiones y conteo

```cpp
// Suposiciones: SAT bajo literales DIMACS (se revierten tras la llamada).
satx::engine e;
const satx::lit_t a = e.add_variable(), b = e.add_variable();
e.add_clause({a, b});                                  // (a ∨ b)
assert(!satx::solver::solve(e, {-a, -b}).has_value()); // ¬a ∧ ¬b → UNSAT
assert(satx::solver::solve(e, {-a}).has_value());      // ¬a → b

// Opciones del kernel + estadísticas (CHB; instancias grandes).
satx::solver::options opt;
opt.heuristic_mode = 1;                                // 0 = VSIDS (default), 1 = CHB
satx::solver::stats st;
auto m = satx::solver::solve(e, opt, &st);
assert(st.conflicts >= 0 && st.clauses > 0);

// Sesión incremental: un solo handle CDCL (conserva cláusulas aprendidas).
satx::solver::session s{e};
assert(s.solve().has_value());
assert(!s.solve({-a, -b}).has_value());
assert(s.solve().has_value());                         // suposiciones revertidas

// Conteo de modelos por cláusulas de bloqueo.
int count = 0;
while (auto m2 = satx::solver::solve(e)) {
    ++count;
    e.add_clause({m2->get(a) ? -a : a, m2->get(b) ? -b : b});  // ¬(modelo)
}
assert(count == 3);                                    // (1,1), (1,0), (0,1)
```

---

## 13. Referencia rápida

### Namespaces públicos

| Namespace | Contenido |
|---|---|
| `satx` | fachada: `engine`, `complex<W,F>`, `lit_t`, operadores, `conj`, `neg`, `real`, `imag`, `eq_lit`, `abs_sq`, `pow`, `root_cbe`, `lt_lit`, `le_lit` |
| `satx::core` | `lit_t`, `true_lit`, `false_lit`, `var_of`, `neg`, `sign`, `is_constant`, `clause`, `cnf` |
| `satx::gates` | `or2 and2 xor2 mux2 fas fac`; `rail rca rca_carry rcs pmul sext zext shl shr sra not_rails and_rails or_rails xor_rails eq ule sle ult slt mux reduce_or reduce_and all_constant` |
| `satx::num` | `complex<W,F>`, `negabinary<W>`, `nb_min`, `nb_max`, `nb_exact_lit`, operadores, `fixed_point::*` |
| `satx::solver` | `result { sat, unsat }`, `options`, `stats`, `model`, `solve` (budget/opciones/suposiciones), `session` |
| `satx::quantum` | `cx`, `mk`, `qgate2`, `qgate4`, `free_gate2/4`, catálogo de compuertas, `qstate`, `qcircuit`, `random_circuit`, `forward_signal`, `otoc_echo`, `constrain_unitary2` |

### Firma rápida de lo esencial

```cpp
// motor
satx::engine e;
auto v  = e.add_variable();
e.add_clause({v, -3});            e.add_unit(v);
bool u = e.unsat();

// números
satx::complex<16, 4> c{1.5, -2.0};                  // constante
satx::complex<16, 4> x{e};                          // incógnita (explicit)
auto z = x + c;   auto p = x * c;   auto q = x / c;  // circuitos
auto k = satx::conj(x);   auto n = satx::neg(x);
auto rp = satx::real(x);  auto ip = satx::imag(x);
auto l = satx::eq_lit(e, x, c);                     // literal de igualdad
e.add_unit(l);                                      // lo impones tú
auto a2 = satx::abs_sq(x);   auto pw = satx::pow(x, 3);   auto rt = satx::root_cbe(x, 2);

// solver
auto m = satx::solver::solve(e);                    // std::expected<model, result>
if (m) { auto val = x.value(*m); }                  // std::complex<double>
auto m2 = satx::solver::solve(e, {-2});             // con suposiciones
satx::solver::options opt; opt.heuristic_mode = 1;  // CHB
satx::solver::session s{e};                         // sesión incremental

// cuántico
using namespace satx::quantum;
qstate<24, 12> s{2};
s.apply1(0, h2<24, 12>());
s.apply2(0, 1, cnot4<24, 12>());
auto ez = s.expect_z(0);
```

### Buenas prácticas

1. **Un `engine` por problema.** No es copiable; pásalo por referencia y no mezcles
   símbolos de motores distintos.
2. **Impón restricciones con `add_unit(eq_lit(...))`**; no uses `operator==` sobre
   símbolos (lanza).
3. **Elige `W`/`F` según rango y resolución**: rango NB de la tabla §3.3, resolución
   `2^(−F)`, y recuerda que `F <= W` y `W <= 60`.
4. **Desbordes:** add/sub envuelven mod 2^W en ambas rutas (ADR-011); `mul` con
   desborde de la caja NB → UNSAT (simbólico) / excepción (concreto); división por
   cero simbólica o cociente no representable → UNSAT; constantes fuera de rango →
   excepción (o `from_raw_wrap`).
5. **Comprueba `e.unsat()`** tras construir restricciones contradictorias para
   cortocircuitar el solver.
6. **En ruta simbólica cuántica**, prefieres `W`/`F` pequeños para aprendizaje (los
   circuitos crecen O(W²) por producto); `W=8, F=4` es un buen punto de partida.
7. **Enumeración y optimización:** para contar modelos añade cláusulas de bloqueo
   (§12.5); para instancias grandes prueba la heurística CHB
   (`options::heuristic_mode = 1`) — en circuitos CBE la diferencia puede ser de
   órdenes de magnitud (`examples/mandelbrot_escape.cpp`).

---

Para el detalle normativo de cada comportamiento (codificación negabinaria, conversiones
NB↔TC, políticas de escala y redondeo, ADR), consulta
[`architecture.md`](architecture.md).

