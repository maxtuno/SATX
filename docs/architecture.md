# satx — Arquitectura de Números Complejo Binario Entrelazado CBE

**Formato numérico propio:** los números de este sistema usan el formato **CBE
(Complejo Binario Entrelazado)**, de autoría de Oscar Riveros (2026, §9.0); no es un
formato negabinario clásico ni ninguna representación numérica previa, estándar o
patentada.

**Copyright (c) 2026 Oscar Riveros. Todos los derechos reservados.** Licencia dual: Apache-2.0 para uso personal; portes a otros lenguajes requieren licencia comercial con autorización expresa del autor (ver LICENSE.txt).

**Estado:** Documento fundacional — Etapa 1: encoding CBE de complejos, las cuatro
operaciones básicas y las operaciones derivadas de la Etapa 1 (unarias, relacionales y de
acceso a rieles — §10.6); **Etapa 2 entregada**: `abs_sq`, comparación lexicográfica
(`lt_lit`/`le_lit`), `pow` con exponente entero (square & multiply) y `root_cbe` (§10.7).
**Revisión ADR-011 entregada**: canonicalización del datapath (todo resultado simbólico se
almacena como patrón TC decodificado a la caja NB), recorte hacia cero exacto en `mul`
(ADR-012), alineación de escala exacta en comparaciones (ADR-013), recurrencia TC→NB
corregida (ADR-006 revisado), guardas de `fixed_point` y extensión del puente Kerberos
(opciones, estadísticas, suposiciones, sesiones — §11).

**Ýmbito:** este documento es la especificación normativa del núcleo numérico de la librería
autónoma `satx` para **C++26**: autocontenida (biblioteca estándar + el kernel SAT de
**Kerberos** — `src/kerberos/slime.c`, parte integral del propio producto, compilado una
sola vez en la biblioteca estática `satx_kerberos`), sin solvers de terceros, sin puentes
a Python.

No es un port literal del `satx` Python: es una **reimplementación** con arquitectura propia,
adecuada a C++26 (`constexpr`, concepts, `std::expected`, spans/ranges).

---

## 1. Propósito

1. Definir la representación canónica de los **números del formato CBE** (*Complejo
   Binario Entrelazado*, §9.0): los dos rieles de enteros en base −2 con escala
   `2^F` que forman el número complejo.
2. Especificar las **cuatro operaciones básicas** — suma, resta, multiplicación, división —
   tanto en ruta **concreta** (plegado de constantes, `constexpr`) como en ruta
   **simbólica** (circuitos booleanos que se compilan a CNF).
3. Fijar una arquitectura en capas que permita extender el sistema a exponenciación,
   logaritmo, trigonométricas y demás trascendentales (CORDIC), y luego a las capas
   superiores (ISA CCMASM, kernel, monedas/`#SAT`, QBF), sin rediseñar el núcleo.

## 2. Extracción del sistema Python (fuentes revisadas)

| Módulo Python | Rol esencial | Destino en C++26 |
|---|---|---|
| `satx/alu.py` (`ALU`) | Motor: asignación de variables, cláusulas CNF, gates booleanos y de bit-vector (RCA, RCS, PM, comparadores) | `satx::engine` + `satx::gates` |
| `satx/unit.py` (`Unit`) | Bit-vector aritmético envolvente sobre bloques de literales | `satx::gates::rail` (vector de literales LSB-first) |
| `satx/cc/cbe.py` (`NegabinaryConverter`) | Codec base −2 y conversiones NB ↔ TC (two's complement) | `satx::num::negabinary` |
| `satx/cc/cbe.py` (`CBE`) | Complejo: dos rieles negabinarios de W bits, escala 2^F | `satx::num::complex<W,F>` |
| `satx/_cbe_api.py` (`CBE`) | Variante pública del complejo con rieles TC directos | Ruta intermedia TC del datapath (no es tipo público) |
| `satx/cordic.py`, `satx/cc/cordic.py` | Trascendentales por CORDIC (circular e hiperbólico) | `satx::num::cordic` (Etapa 3) |
| `src/kerberos/slime.c` (SLIME) | Kernel SAT: CDCL industrial con API C embebida (`slime_sat_handle_*`); el despachador `kerberos.c` (SLIME/BASILISK/PIXIE/WMIBO) queda disponible para uso CLI | `satx::solver::kerberos` (puente) + kernel integrado en `src/kerberos/` (biblioteca `satx_kerberos`, sin third_party) |
| `satx/fixed.py` (`Fixed`) | Punto fijo decimal (escala 10^k) — concepto distinto de CBE | `satx::num::fixed` (futuro, no confundir con 2^F) |
| `satx/cc/ccmasm.py`, `satx/kernel.py` | ISA CCMASM + ciclo fetch/decode/execute/writeback | `satx::ccm` (Etapa 5) |

**Decisión de diseño (ADR-002, revisado por ADR-011):** el núcleo adopta el modelo de
`satx/cc/cbe.py` (representación canónica negabinaria) y **no** el de `satx/_cbe_api.py`
(TC directo). La aritmética simbólica se ejecuta internamente sobre patrones TC mod 2^W —
como hace la referencia — porque los circuitos TC (RCA/RCS/PM) son simples y compactos.
Los **resultados** de las operaciones se almacenan como patrones TC de W bits cuya
decodificación canónica es el pliegue `wrap_nb` (el representante de la caja NB de su
clase mod 2^W): la decodificación es así biyectiva y coherente con el wrap de add/sub.
La conversión NB ↔ TC vive en la frontera (construcción de constantes y preparación de
operandos).

## 3. Principios de diseño

- **Sin dependencias externas:** std de C++26 más el kernel CDCL **SLIME de Kerberos**
  y sus backends (fuentes C del propio repositorio en `src/kerberos/`, compilados una
  sola vez como biblioteca estática `satx_kerberos`). No hay solvers de terceros ni
  puentes a Python.
- **Compiladores:** GCC/Clang (la ruta concreta usa `__int128` para productos y divisiones
  exactas sin desbordes intermedios).
- **Sin estado global:** el `engine` (asignador de variables + base de cláusulas) es un
  objeto explícito que se pasa por referencia. Mejora sobre el `csp` global de Python.
- **Plantillas sobre W/F:** `complex<W,F>` con `W`, `F` como parámetros no-tipo
  `std::size_t`; toda la aritmética de constantes es `constexpr`. Una variante dinámica
  (W, F en tiempo de ejecución) se añadirá como envoltorio en la Etapa 5 para la ISA.
- **Ruta dual:** cada operación tiene (a) plegado concreto cuando ambos operandos son
  constantes y (b) construcción de circuito cuando intervienen variables. El tipo público
  oculta la dualidad, igual que `CBE` en Python.
- **Correctitud matemática primero:** donde la referencia Python contenga simplificaciones
  erróneas, la especificación C++ manda (véanse §7.3 y los ADR-006/ADR-007/ADR-011..015).

## 4. Capas de la arquitectura

```
+--------------------------------------------------------------+
| API pública: namespace satx (operadores, fábricas, decodif.) |
+--------------------------------------------------------------+
| num::complex<W,F>  — números negabinarios complejos          |
|   · ruta concreta (constexpr)   · ruta simbólica (circuitos) |
+--------------------------------------------------------------+
| num::negabinary — codec NB; num::fixed_point — escala 2^F    |
+--------------------------------------------------------------+
| gates::bitvec — rail, RCA, RCS, PM, sext/zext/shr/sra,       |
|   bit a bit, EQ/ULE/SLE/SLT/ULT, rca_carry                   |
+--------------------------------------------------------------+
| gates::primitive — AND/OR/XOR/MUX/FAS/FAC (Tseitin → CNF)    |
+--------------------------------------------------------------+
| core — Lit, Clause, Cnf (arena), engine (variables)          |
+--------------------------------------------------------------+
| solver::kerberos — puente → kernel CDCL SLIME de Kerberos;   |
|   options/stats/suposiciones/sesiones; solver::model         |
+--------------------------------------------------------------+
```

Cada capa depende solo de las inferiores. Las capas futuras (CORDIC, `#SAT`, CCMASM) se
apoyan sobre `gates` y `num` sin tocar `core`. La capa `solver` es la única que enlaza C:
enlaza la biblioteca `satx_kerberos` (fuentes de `src/kerberos/`, compiladas una sola vez)
y usa la API embebida declarada en `src/kerberos/slime_bridge.h`.

## 5. Núcleo booleano (`satx::core`)

### 5.1 Literales y cláusulas

```cpp
namespace satx::core {
  using lit_t = std::int32_t;              // 0 = inválido; ±v = variable v (v >= 1)
  constexpr lit_t true_lit  =  1;          // constante VERDADERO (variable 1 fijada)
  constexpr lit_t false_lit = -1;          // constante FALSO

  constexpr std::int32_t var_of(lit_t l) noexcept { return std::abs(l); }
  constexpr lit_t neg(lit_t l) noexcept { return -l; }
  constexpr bool sign(lit_t l) noexcept { return l > 0; }
  constexpr bool is_constant(lit_t l) noexcept { return l == true_lit || l == false_lit; }

  struct clause { std::vector<lit_t> lits; };   // disyunción de literales
}
```

Invariantes (heredados del ALU Python, corregidos por ADR-007 y ADR-011):

- La variable 1 es la constante VERDADERO; el `engine` añade la cláusula unitaria
  `[true_lit]` (es decir, `[+1]`) al construirse. (Nota: la referencia Python añade
  `[-true_lit]`; su convención `true`/`false` está invertida y la especificación C++
  la corrige — ADR-007.)
- Una cláusula se normaliza: ordenada por `|lit|`, sin duplicados, tautologías eliminadas.
- Cláusulas idénticas se deduplican por hash de `(lits)`, como `ALU.add_block`.
- La cláusula vacía `[]` representa UNSAT; la unidad `[false_lit]` (`¬VERDADERO`) también
  marca UNSAT de inmediato (ADR-011).
- Literales inválidos (0 o `INT32_MIN`) se rechazan con `std::invalid_argument` al añadir
  la cláusula (ADR-011: antes llegaban al kernel y fallaban tarde y confusamente).

### 5.2 Motor (`engine`)

```cpp
namespace satx {
  class engine {
  public:
    [[nodiscard]] core::lit_t add_variable();                    // nueva variable booleana
    void add_clause(std::span<const core::lit_t> lits);          // disyunción
    void add_clause(std::initializer_list<core::lit_t> lits);    // comodidad sintáctica
    void add_unit(core::lit_t l);

    std::size_t variable_count() const noexcept;
    std::size_t clause_count()   const noexcept;
    bool unsat() const noexcept;                                 // cláusula vacía o [¬1]
    core::cnf const& formula() const noexcept;                   // acceso de solo lectura

    // política de alineación de anchos en gates::bitvec (ADR-011: implementada)
    enum class width_policy { sign_extend, truncate };           // default: sign_extend
    void set_width_policy(width_policy p) noexcept;
    [[nodiscard]] width_policy get_width_policy() const noexcept;

  private:
    core::cnf clauses_;                                          // arena + índice de literales
  };
}
```

- `add_clause` normaliza y deduplica (hash por `(lits)`) como `ALU.add_block`.
- `unsat()`/`formula()`: O(1). `formula()` es el acceso de solo lectura que el puente
  `solver::kerberos` vuelca a la C API de SLIME (§11); `unsat()` cortocircuita la
  llamada al kernel.
- El motor NO resuelve; la resolución es responsabilidad de `solver::kerberos` (puente
  sobre el kernel CDCL SLIME de Kerberos). Separar construcción de resolución es la
  diferencia estructural clave con el ALU Python.
- **ADR-011:** `const_policy` fue eliminado del motor (era un interruptor muerto): la
  codificación de constantes es siempre estricta (`std::out_of_range` fuera de la caja NB)
  y el pliegue wrap se obtiene con la fábrica `complex::from_raw_wrap`.
  `width_policy::truncate` recorta los operandos al ancho menor en `rca`/`rcs`/`pmul`/
  `eq`/`ule`/`sle`/`mux` (la capa `num` siempre pre-alinea anchos, así que la política
  solo afecta al uso directo de `gates::bitvec`).

## 6. Gates (Tseitin → CNF)

Cada gate crea una variable de salida `o` y emite las cláusulas de su tabla de verdad.
Catálogo mínimo (idéntico al `ALU` Python, recodificado en C++):

| Gate | Firma | Cláusulas |
|---|---|---|
| `gate_or2`  | `lit_t or2(engine&, lit_t a, lit_t b)` | 4 |
| `gate_and2` | `lit_t and2(engine&, lit_t a, lit_t b)` | 4 |
| `gate_xor2` | `lit_t xor2(engine&, lit_t a, lit_t b)` | 4 |
| `gate_mux2` | `lit_t mux2(engine&, lit_t s, lit_t a, lit_t b)` | 4 |
| `gate_fas`  | `lit_t fas(engine&, lit_t a, lit_t b, lit_t ci)` — bit de suma | 8 |
| `gate_fac`  | `lit_t fac(engine&, lit_t a, lit_t b, lit_t ci)` — bit de acarreo | 6 |

`fas`/`fac` son las células del sumador completo y constituyen la base de toda la
aritmética. Un bit constante 0/1 se materializa como `false_lit`/`true_lit` (no consume
variables). Todos los gates pliegan operandos constantes (ADR-010): por ejemplo,
`and2(x, false_lit) == false_lit` sin emitir cláusulas, y `rca`/`rcs` pliegan rieles
totalmente constantes por aritmética directa (con `__int128` para anchos ≤ 127 en
`rca`/`rca_carry` y `uint64_t` para anchos ≤ 64 en `pmul` — ADR-011: sin UB de
desplazamientos ≥ 64).

### 6.1 Bit-vectors (`gates::bitvec`)

Un **riel** es `std::vector<lit_t>` en orden **LSB-first** (posición 0 = bit 0), igual que
los bloques de Python.

```cpp
namespace satx::gates {
  using rail = std::vector<core::lit_t>;

  rail rca(engine&, rail a, rail b, core::lit_t cin = core::false_lit); // suma (ripple-carry)
  std::pair<rail, core::lit_t> rca_carry(engine&, rail a, rail b,
                                         core::lit_t cin = core::false_lit); // suma + acarreo
  rail rcs(engine&, rail a, rail b);                                    // resta: a + ¬b + 1
  rail pmul(engine&, rail a, rail b);                                   // productos parciales
  rail sext(rail a, std::size_t w);                                     // extensión de signo
  rail zext(rail a, std::size_t w);                                     // extensión con ceros
  rail shl(rail a, std::size_t k);                                      // << k (wrap), sin cláusulas
  rail shr(rail a, std::size_t k);                                      // >> k lógico, sin cláusulas
  rail sra(rail a, std::size_t k);                                      // >> k aritmético, sin cláusulas
  rail not_rails(rail a);                                               // negación bit a bit, sin cláusulas
  rail and_rails(engine&, rail a, rail b);                              // AND bit a bit
  rail or_rails(engine&, rail a, rail b);                               // OR bit a bit
  rail xor_rails(engine&, rail a, rail b);                              // XOR bit a bit
  core::lit_t eq(engine&, rail a, rail b);                              // igualdad bit a bit
  core::lit_t ule(engine&, rail a, rail b);                             // <= sin signo
  core::lit_t sle(engine&, rail a, rail b);                             // <= con signo
  core::lit_t ult(engine&, rail a, rail b);                             // < sin signo
  core::lit_t slt(engine&, rail a, rail b);                             // < con signo
  rail mux(engine&, core::lit_t s, rail a, rail b);
  bool all_constant(rail const& a);                                     // riel 100% constante
  core::lit_t reduce_or(engine&, rail a);                               // OR reducido (≠ 0)
  core::lit_t reduce_and(engine&, rail a);                              // AND reducido (todo 1)
}
```

- `rca`: W× `fas` + W× `fac` emitidos (el primero se pliega con el acarreo
  constante); el acarreo final es descartable (o accesible vía `rca_carry`).
- `rcs`: complemento de `b` con acarreo de entrada `true_lit` (suma en complemento a dos).
- `pmul`: multiplicador por productos parciales (`AND` de rieles) y reducción con `rca`;
  complejidad O(W²) en variables/cláusulas.
- `sext`: replica el bit más significativo (semántica con signo); `zext` rellena con
  `false_lit` (semántica sin signo). `shr`/`sra`/`not_rails` re-enrutan literales sin
  cláusulas. `slt`/`ult` son `¬sle(b,a)`/`¬ule(b,a)` (el orden es total).
- `shl`: desplazamiento a la izquierda con wrap dentro del ancho; no emite cláusulas
  (re-enrutado O(W) de literales). Implementa la alineación de escala `·2^(F−Fa)` de
  add/sub (§10, ADR-008).
- `all_constant`: predicado O(W) de host; habilita el plegado aritmético directo de
  `rca`/`rcs`/`pmul` cuando ambos rieles son constantes (ADR-010).
- `eq`: (W−1)× `and2` sobre `¬xor2` por bit; O(W). `ule`/`sle`: cadenas comparadoras
  MSB-first; O(W). `reduce_or`: cadena de (W−1)× `or2`; O(W); es la restricción
  «distinto de cero» del divisor en `operator/` (§10.4).
- Los operadores de ancho mixto aplican la política del motor (`width_policy`):
  **sign extension** al ancho máximo por defecto (el comportamiento requerido por CBE),
  o truncado al ancho menor con `width_policy::truncate`.

## 7. Encoding negabinario (`satx::num::negabinary`)

### 7.1 Definición

Un entero `n` se representa con `W` dígitos binarios `d_i ∈ {0,1}` tales que

```
n = Σ_{i=0}^{W-1} d_i · (−2)^i
```

Propiedades:

- La representación es **única** para todo `n` en el rango y cubre exactamente `2^W`
  enteros distintos.
- El rango es **asimétrico**:

```
min_NB(W) = −2·(4^⌊W/2⌋ − 1)/3          (suma de potencias de posición impar)
max_NB(W) =    (4^⌈W/2⌉ − 1)/3          (suma de potencias de posición par)
```

| W | rango NB | rango TC (referencia) |
|---|---|---|
| 8  | [−170, 85]      | [−128, 127] |
| 12 | [−2730, 1365]   | [−2048, 2047] |
| 16 | [−43690, 21845] | [−32768, 32767] |
| 32 | [−2863311530, 1431655765] | [−2^31, 2^31−1] |

Nótese `max_NB + |min_NB| = 2^W − 1`.

### 7.2 Codec (ruta concreta, `constexpr`)

```cpp
namespace satx::num {
  constexpr std::int64_t nb_min(std::size_t w) noexcept;   // −2·(4^⌊w/2⌋−1)/3 (w ≤ 60)
  constexpr std::int64_t nb_max(std::size_t w) noexcept;   //  (4^⌈w/2⌉−1)/3  (w ≤ 60)

  template<std::size_t W>
    requires (W >= 2 && W <= 60)
  struct negabinary {
    std::array<std::uint8_t, W> digits{};   // LSB-first

    static constexpr negabinary encode(std::int64_t n);   // división repetida por −2
    constexpr std::int64_t decode() const noexcept;       // Σ d_i·(−2)^i
  };
}
```

Algoritmo de codificación (división con resto normalizado a {0,1}):

```
for i in 0..W-1:
    q, r = divmod(n, -2)      # división truncada, r ∈ {0, -1}
    if r < 0: q += 1; r += 2  # r ∈ {0, 1}
    digits[i] = r
    n = q
# validación: n == 0 al final  ⟺  el valor cabe en W bits NB
```

La codificación de constantes comprueba `min_NB(W) ≤ v ≤ max_NB(W)`; fuera de rango:
`std::out_of_range` (ruta concreta, siempre estricta — ADR-011). La variante
`complex::from_raw_wrap` pliega mod 2^W al rango NB.

### 7.3 Conversiones NB ↔ TC (circuitos)

Dado el riel negabinario `d` y el riel TC `t` (ambos de `W` bits, LSB-first):

**NB → TC** (patrón mod 2^W, sin estados intermedios):

```
pos[i] = d[i]  si i es par,  false_lit en otro caso
neg[i] = d[i]  si i es impar, false_lit en otro caso
t = rcs(pos, neg)        # t = pos − neg
```

Justificación: `(−2)^i = +2^i` para `i` par y `−2^i` para `i` impar, por lo que
`n = Σ d_i·2^i (par) − Σ d_i·2^i (impar)`. **Nota (ADR-011):** el patrón devuelto es
`n mod 2^W`; los dígitos NB en sí NO son el patrón mod 2^W (solo coinciden mod 2).

**TC → NB** (ADR-006 revisado — la forma anterior con acarreo de un bit
`c_prev = (i par) ? (t_i ∧ c) : (t_i ∨ c)` y condición «exacto ⟺ c_final == t[W−1]»
era **incorrecta**: p. ej. en W=3 daba falsos positivos/negativos). Recurrencia
correcta con resto completo:

```
R_0 = patrón (mod 2^W)
for i in 0..W-1:
    d[i] = R_i mod 2                # bit 0
    R_{i+1} = (R_i >> 1) + (i impar ? d[i] : 0)
```

Produce **siempre** los dígitos NB del representante canónico de la caja (`wrap_nb`
del patrón), para cualquier patrón de W bits. Coste O(W²); solo se usa en los
accesores `real_rail`/`imag_rail` de objetos TC. La **exactitud** (el valor con signo
está en la caja NB ⟺ `min_NB(W) ≤ v ≤ max_NB(W)`) se implementa con comparadores en
W+1 bits y está disponible como `nb_exact_lit` (ADR-011); el test §14.2 verifica la
identidad y la exactitud sobre el rango completo.

**Conversión exacta NB → TC con signo:** para la preparación de operandos de `mul`,
comparaciones y división se usa el ancho `W+1` (la caja NB cabe en el rango con signo
de W+1 bits): cero-extender los dígitos NB a W+1 y aplicar NB→TC. Para objetos TC se
usa `tc_to_signed` (comparador contra `max_NB` + resta condicional de 2^W).

## 8. Punto fijo (escala 2^F)

El valor numérico de un riel con `F` bits fraccionarios es `raw / 2^F`, con `raw` entero
en rango NB (constantes) o como patrón TC decodificado a la caja NB (resultados de
circuitos, ADR-011). La escala es **potencia de dos** (no la escala decimal 10^k de
`fixed.py`, que es un tipo distinto y posterior).

```cpp
namespace satx::num::fixed_point {
  constexpr std::int64_t scale(std::size_t f);                     // 2^f
  constexpr std::int64_t to_raw(double x, std::size_t f);          // round(x · 2^f)
  constexpr double from_raw(std::int64_t raw, std::size_t f);      // raw / 2^f
  constexpr std::int64_t trunc_shr(std::int64_t v, std::size_t k);       // >> truncado a 0
  constexpr std::int64_t trunc_shr_i128(__int128 v, std::size_t k);      // ídem, 128 bits
  constexpr std::int64_t round_scaled_div(__int128 n, __int128 den,
                                          std::size_t k);          // round(n·2^k / den)
}
```

- Codificación de una constante real `x`: `raw = round(x · 2^F)` con **redondeo
  half-away-from-zero** (`std::round`). ADR-003: la referencia Python usa `round()`
  (half-even); se documenta la divergencia como decisión deliberada.
- Decodificación: `x = raw / 2^F` exacto (división de enteros → `double`/`long double`).
- Resolución: `2^(−F)`.
- `trunc_shr`/`trunc_shr_i128`: desplazamiento aritmético con **truncado hacia cero**
  (el `>>` nativo trunca hacia −∞ en negativos). ADR-003/ADR-012: es el recorte simétrico
  que aplica `mul` en **ambas** rutas (§10.3).
- `round_scaled_div`: `round(n · 2^k / den)` sin desbordes intermedios — numerador de
  128 bits expandido a 256 y división larga bit a bit (241 iteraciones, O(1)).
  `den <= 0` → `std::domain_error`; `k > 120` → `std::domain_error`;
  `|resultado| >= 2^62` o `n·2^k` con más de 241 bits → `std::overflow_error`
  (ADR-011: antes los casos extremos devolvían cocientes truncados silenciosamente y
  había UB para k > 128). Es la primitiva de la ruta concreta de `operator/` (§10.4).
- `to_raw` lanza `std::out_of_range` si `x · 2^F` no cabe en `int64` (no finito incluido) —
  ADR-011: sin UB en la conversión float→int64.

## 9. El número complejo CBE (`satx::num::complex<W,F>`)

### 9.0 Nota de autoría: el formato CBE(W,F) — Complejo Binario Entrelazado

**El formato numérico que usa este sistema — CBE, *Complejo Binario Entrelazado* —
es de autoría propia de Oscar Riveros (2026)**, definido en el documento
«Modelo Unificado de Cómputo Clónico» (sección 10).
No procede de ningún estándar, biblioteca ni trabajo previo: **no es la
representación clásica de números complejos**; es el formato original de la librería,
diseñado para que la aritmética compleja sea expresable como circuitos booleanos
resolubles por SAT.

**Definición (CBE(W,F)).** Una palabra de `2W` bits `Z[2W−1:0]` representa un número
complejo `z = a + i·b` con **entrelazado por cableado**: los **bits pares** alimentan
el carril real y los **bits impares** el carril imaginario:

```
Z[2k]   = R[k]     (carril real)
Z[2k+1] = I[k]     (carril imaginario)      k = 0..W−1
```

Cada carril usa **base −2** con `F` posiciones fraccionarias; el exponente de la
posición `k` es `e(k) = k − F`:

```
Re(Z) = Σ_{k=0}^{W−1} R[k]·(−2)^(k−F)
Im(Z) = Σ_{k=0}^{W−1} I[k]·(−2)^(k−F)
Z     = Re(Z) + i·Im(Z)
```

Forma equivalente usada por la API: `z = (re + i·im) · 2^(−F)` con `re, im ∈ ℤ`
codificados en **base −2 (negabinaria)** con `W` dígitos binarios `d_i ∈ {0,1}` cada
uno (§7.1):

```
re = Σ_{i=0}^{W−1} d_i · (−2)^i        (y análogo para im)
```

El paso numérico mínimo es `2^(−F)`. `F` es la escala fraccionaria fija (punto fijo
binario, §8). La palabra física de `2W` bits está **entrelazada**: `Z[2k] = R[k]`,
`Z[2k+1] = I[k]` (la API la separa en los dos carriles con `DEINTERLEAVE` y la
recompone con `INTERLEAVE`).

**Por qué base −2.** La base −2 representa enteros con signo **sin bit de signo
separado**: los valores negativos surgen de las potencias de índice impar, que son
negativas. Para cada entero del rango la representación es **única** —no hay formas
redundantes como el `−0` del complemento a dos— y el rango es asimétrico:

```
min_NB(W) = −2·(4^⌊W/2⌋ − 1)/3 ,   max_NB(W) = (4^⌈W/2⌉ − 1)/3
```

con `max_NB + |min_NB| = 2^W − 1` (§7.1). Ejemplos (W = 4, dígitos MSB-first
`d₃d₂d₝d₀`): `−1 = 0011` (ya que `1·(−2)¹ + 1·(−2)❰ = −2 + 1`), `+1 = 0001`,
`i·2^(−F)` es el riel imaginario `0001`.

**Propiedad estructural (biyección).** El mapa

```
bits (2W) ⟼ (re, im) ⟼ z = (re + i·im) · 2^(−F)
```

es una **biyección** entre las `2^(2W)` palabras binarias y la caja del retículo

```
CBE(W,F) = { (a + bi)·2^(−F) : a, b ∈ [min_NB(W), max_NB(W)] } ⊂ ℂ
```

Toda palabra de bits es un número complejo válido y todo número de la caja tiene
exactamente una palabra; no hay patrones inválidos, huecos ni palabras redundantes.
La cardinalidad es exactamente `2^(2W) = |ℤ_NB|²`.

**Consecuencias prácticas.** Como la representación es biyectiva y la escala es
potencia de dos: (a) las cuatro operaciones aritméticas se compilan a circuitos de
bit-vector (datapath interno sobre patrones TC mod 2^W, con conversiones NB↔TC
lineales y exactas, §7.3 y ADR-002/ADR-011); (b) la igualdad de dos números es
igualdad bit a bit sobre la representación exacta alineada; (c) el signo, la fase y la
interferencia de amplitudes son aritmética exacta sobre enteros, no convenciones ad hoc.
El formato permite, además, tratar cada coeficiente como **variable booleana libre** en
la ruta simbólica: un complejo desconocido es una incógnita de `2W` bits con dominio
exacto `CBE(W,F)`.

### 9.1 Representación

```cpp
template<std::size_t W, std::size_t F>
  requires (W >= 2 && W <= 60 && F <= W)
class complex {
public:
  enum class kind { concrete, nb, tc };             // representación física del objeto
  static constexpr std::size_t width = W;           // y fractional = F

  // — ruta concreta (constante plegada) —
  constexpr complex();                              // (0, 0)
  constexpr complex(std::int64_t re_raw, std::int64_t im_raw);  // valores ya escalados
  template<typename A, typename B>                  // A, B floating_point
    requires (std::floating_point<A> && std::floating_point<B>)
  constexpr complex(A re, B im);                    // codifica en NB con escala 2^F

  // — ruta simbólica (variable libre, rieles NB) —
  explicit complex(engine& e);                      // 2W variables frescas

  static constexpr complex i_unit(engine& e);       // (0, 2^F)
  static constexpr complex one(engine& e);          // (2^F, 0)
  static constexpr complex zero(engine& e);         // (0, 0)
  static constexpr complex from_float(double re, double im);
  static constexpr complex from_raw(std::int64_t re_raw, std::int64_t im_raw);
  static constexpr complex from_raw_wrap(std::int64_t re_raw, std::int64_t im_raw);
  static complex from_tc_rails(engine&, gates::rail re, gates::rail im); // patrón TC
  static complex from_nb_rails(engine&, std::array<lit_t,W> re,
                               std::array<lit_t,W> im);                  // dígitos NB

  // — acceso e introspección —
  constexpr kind representation() const noexcept;   // concrete | nb | tc
  constexpr bool is_concrete() const noexcept;
  engine* engine_of() const noexcept;               // nullptr en constantes
  constexpr std::int64_t re_raw() const;            // solo constantes (si no: logic_error)
  constexpr std::int64_t im_raw() const;
  std::array<core::lit_t, W> re_pattern() const noexcept;  // patrón almacenado, sin circuitos
  std::array<core::lit_t, W> im_pattern() const noexcept;
  std::array<core::lit_t, W> real_rail(engine&) const;  // riel NB canónico (§7.3 bajo demanda)
  std::array<core::lit_t, W> imag_rail(engine&) const;
  std::array<core::lit_t, W> tc_real_rail(engine&) const; // patrón TC del datapath
  std::array<core::lit_t, W> tc_imag_rail(engine&) const;

  // pliegue mod 2^W al rango NB (representante canónico de la caja)
  static constexpr std::int64_t wrap_nb(std::int64_t v) noexcept;

  // decodificación post-solve
  std::complex<double> value(model const&) const;   // NB directo o patrón TC → wrap_nb (÷ 2^F)
};
```

El límite `W <= 60` garantiza que el rango NB completo quepa en `std::int64_t`
(`4^⌈W/2⌉ <= 2^60`) y que las máscaras de `2^W` bits quepan en `std::uint64_t`.

Definición semántica:

```
z = (re + i·im) / 2^F,   re, im enteros
```

- **Formato físico (canónico):** palabra de `2W` bits **entrelazada** (§9.0): los bits
  pares `Z[2k]` son el riel real NB `R[k]` y los bits impares `Z[2k+1]` el riel
  imaginario NB `I[k]`. (`bits == 2W`, como en la referencia.)
- **Formato del datapath (ADR-011):** internamente los circuitos aritméticos operan
  sobre **patrones TC mod 2^W** (`_tc_real`/`_tc_imag` en la referencia). Todo resultado
  de operación simbólica se almacena como `kind::tc` (patrón) y `value(model)` lo
  decodifica con `wrap_nb`: el representante canónico de la caja NB de su clase mod 2^W.
  Para add/sub/neg/conj esto implementa el wrap documentado; para mul/div las
  restricciones de desborde garantizan que el valor esté en la caja y la decodificación
  sea exacta. `re_pattern()`/`im_pattern()` exponen el patrón almacenado sin circuitos.
- `value(model)` reconstruye cada riel desde el modelo: constantes → directo; `nb` →
  decodificación NB; `tc` → patrón → `wrap_nb`; y divide por `2^F`.

### 9.2 Fábricas y conversiones

- `complex::from_float(re, im)`: ruta concreta con validación de rango NB (y de rango
  `int64` — ADR-011).
- `complex::from_raw(re_raw, im_raw)`: valores ya escalados; `from_raw_wrap` pliega
  mod 2^W al rango NB.
- `complex::from_tc_rails(tc_re, tc_im)`: promoción de rieles TC (resultados internos);
  la semántica del patrón es la de `wrap_nb` (ADR-011). Exige anchos exactos
  (`std::invalid_argument` si no — ADR-011).
- `complex::from_nb_rails(re, im)`: rieles NB directos (usada por `real`/`imag`).
- Conversiones §7.3 bajo demanda (miembros `tc_real_rail`/`tc_imag_rail` y
  `real_rail`/`imag_rail`). La exactitud de la conversión TC→NB está disponible como
  `nb_exact_lit` (valor con signo en la caja NB, comparadores en W+1 bits — ADR-006
  revisado), exportada al namespace público `satx::num`.
- Igualdad: `operator==`/`!=` comparan valores y solo operan sobre constantes (ruta
  concreta); la igualdad bit a bit por riel sobre circuitos es `eq_lit(engine&, a, b)`,
  que devuelve un literal (sext y alineación de escala EXACTA — ADR-013). El caso
  `eq_lit(e, z, z)` con el mismo objeto NB pliega a `true_lit` sin circuito (§14.3).

## 10. Operaciones básicas

Convención: `W = max(Wa, Wb)`, `F = max(Fa, Fb)`. Cada operación define dos rutas:
**concreta** (ambos operandos constantes → aritmética `constexpr` del host y
re-codificación) y **simbólica** (circuitos). Revisión ADR-011: add/sub envuelven
mod 2^W en **ambas** rutas (la concreta ya no lanza), y mul/div restringen el
resultado a la caja NB en ambas rutas (la simbólica con comparadores, la concreta con
la validación del constructor).

**Alineación de escala con F mixtos (ADR-008/ADR-013):** la referencia Python usa
`F = max(Fa,Fb)` sin realinear los operandos (defecto latente). La especificación C++
realinea:

- add/sub: el operando con escala menor se desplaza `F − Fa` bits a la izquierda sobre
  el patrón mod 2^W (multiplicar por `2^(F−Fa)`; wrap en circuitos). El resultado
  mod 2^W es exacto porque la suma mod 2^W es un homomorfismo;
- mul: el producto está a escala `2^(Fa+Fb)` y se recorta con `K = min(Fa, Fb)`
  (ventana `[K, K+W)` de los rieles de `2W+2` bits — los operandos entran exactos en
  W+1 bits, ADR-011);
- eq/lt/le: la alineación es **exacta** (sin wrap) sobre el ancho
  `max(Wa+F−Fa, Wb+F−Fb)+1` — ADR-013: con wrap, restricciones del usuario como
  `x == b` se volvían vacuamente satisfacibles cuando el desplazamiento perdía bits;
- div: el cociente está a escala `2^(F+Fb−Fa)` (`K = F + Fb − Fa` en la fórmula exacta).

Con `Fa == Fb` (el caso habitual) todo degenera a las fórmulas clásicas: `>> F` en mul
y `· 2^F` en div.

### 10.1 Suma — `operator+`

```
re = re_a + re_b        (mod 2^W, wrap)
im = im_a + im_b
```

Circuito: patrones mod 2^W (zext al ancho común + `shl` de alineación) y `rca`; el
resultado se almacena como patrón TC (ADR-011). Semántica de desborde: envolvente
(wrap) por defecto, **idéntica en ambas rutas** (la ruta concreta pliega con
`wrap_nb128` — ADR-011; ya no lanza).

### 10.2 Resta — `operator-`

```
re = re_a − re_b ;  im = im_a − im_b        (mod 2^W, wrap)
```

Circuito: `rcs` sobre los patrones; resultado como patrón TC. Misma semántica de
wrap que la suma.

### 10.3 Multiplicación — `operator*`

Con `a = ar + i·ai`, `b = br + i·bi` (todos enteros escalados):

```
P_re = ar·br − ai·bi          (a escala 2^(Fa+Fb))
P_im = ar·bi + ai·br
re = trunc(P_re / 2^K) ;   im = trunc(P_im / 2^K)     (K = min(Fa,Fb); truncado hacia cero)
```

Circuito (estructura de `CBE.mul`, corregida por ADR-011/ADR-012):

1. **Entradas exactas:** cada riel se convierte a TC con signo en `W+1` bits y se
   sext-ea a `2W+2` (el producto de dos valores con signo de W+1 bits cabe exactamente).
2. Cuatro productos: `ac = pmul(ar, br)`, `bd = pmul(ai, bi)`, `ad = pmul(ar, bi)`,
   `bc = pmul(ai, br)` (rieles de `2W+2` bits).
3. Combinación: `re_full = rcs(ac, bd)`, `im_full = rca(ad, bc)`.
4. **Recorte hacia cero (ADR-012):** `trunc(P/2^K) = (P + corr) >> K` (aritmético) con
   `corr = (signo de P) ? 2^K − 1 : 0`. El recorte de la ventana `[K, K+W)` a secas es
   *floor*, no truncado; la corrección iguala ambas rutas (la concreta usa
   `trunc_shr_i128`).
5. **Chequeo de desborde (ADR-011):** el cociente truncado `Q` debe caber en la caja
   NB. (a) los bits por encima de la ventana con signo de W+1 bits deben igualar al
   signo — `Q ∈ [−2^W, 2^W−1]`; (b) el valor completo `Q` (bits `[K, K+W+1)`) debe
   estar en `[min_NB(W), max_NB(W)]` (comparadores con signo contra constantes).
   En ruta simbólica: restricciones insatisfacibles si el resultado desborda; en ruta
   concreta: `std::out_of_range` del constructor. La ventana sola (Q mod 2^W) NO basta:
   admite alias fuera de caja.

El resultado se almacena como patrón TC de `W` bits (ADR-011), que decodifica a la
caja NB por la restricción de desborde.

### 10.4 División — `operator/`

Ruta concreta (ambos operandos constantes): fórmula exacta

```
(a + bi) / (c + di) = ((ac + bd) + i(bc − ad)) / (c² + d²)
```

escalada y redondeada a `2^F` con `round_scaled_div(n, d, K)` — división larga exacta
sin desbordes intermedios (256 bits); división por cero → `std::domain_error`.

Ruta simbólica: se crea un `complex` libre `r` y se impone la identidad

```
mul(r, b) == a        (igualdad bit a bit sobre la representación exacta alineada)
```

más la restricción de divisor no nulo sobre los **rieles nativos** de `b` (sin
alineación de escala — ADR-011: con rieles re-escalados, divisores válidos producían
UNSAT espurio). La división simbólica es **exacta-en-cuadrícula**: cocientes no
representables en la grilla son UNSAT (no se redondean); dividir por cero simbólico
produce UNSAT. La Etapa 3 sustituirá esta ruta por un divisor iterativo
(Newton-Raphson / CORDIC) con redondeo.

### 10.5 Tabla de complejidad de circuitos (por operación)

| Operación | Celdas principales | Coste aproximado |
|---|---|---|
| add/sub | 2× RCA de W bits (+ conversión NB→TC O(W) por operando nb) | O(W) |
| mul | 4× PM de (2W+2)×(2W+2) + 2 combinaciones + recorte y caja | O(W²) |
| div (simbólica) | 1× mul + 2× EQ exactas + divisor ≠ 0 | O(W²) |

### 10.6 Operaciones derivadas de la Etapa 1 (categorías, coste y dependencias)

Además de las cuatro operaciones básicas, la Etapa 1 entrega unarias, relacionales y
miembros de acceso/decodificación. Se agrupan por categoría, con su coste de circuito y
la capa de la que dependen.

**Categoría A — unarias** (dependen de `gates::bitvec` en ruta simbólica y de
`negabinary`/`fixed_point` en ruta concreta):

| Operación | Ruta concreta | Ruta simbólica (circuito) | Coste |
|---|---|---|---|
| `neg(z)` | `wrap_nb(−re_raw, −im_raw)` | `re' = rcs(0, patrón)`, `im' = rcs(0, patrón)` | 2× RCS, O(W) |
| `conj(z)` | `(re_raw, wrap_nb(−im_raw))` | `re' = patrón` (sin circuito), `im' = rcs(0, patrón)` | 1× RCS, O(W) |
| `real(z)` / `imag(z)` | `(re_raw, 0)` / `(im_raw, 0)` | componente pedida + riel cero (`false_lit`), sin circuitos | O(1) |

**Categoría B — relacionales (igualdad):**

| Operación | Ruta concreta | Ruta simbólica | Coste |
|---|---|---|---|
| `eq_lit(engine&, a, b)` | pliega a `true_lit`/`false_lit` comparando `(re,im)` con alineación de escala a `F` común (`__int128`) | `and2(eq(re), eq(im))` sobre rieles exactos al ancho `max(Wa+F−Fa, Wb+F−Fb)+1` (ADR-013) | 2× EQ + 1× AND, O(W) |
| `operator==` / `operator!=` | comparación de valor con `__int128` (misma alineación) | no disponible: lanza `std::logic_error` (usar `eq_lit`) | O(1) host |

Nota de plegado (ADR-010): `gates::eq` sobre rieles idénticos devuelve `true_lit` sin
circuito y sobre rieles 100% constantes distintos devuelve `false_lit`. Además,
`eq_lit(e, z, z)` con el MISMO objeto NB devuelve `true_lit` directamente (§14.3).

**Categoría C — acceso, conversión de rieles y decodificación** (miembros de `complex`;
no consumen variables salvo que se solicite una conversión §7.3):

| Miembro | Función | Dependencia | Coste |
|---|---|---|---|
| `representation()` / `is_concrete()` | estado físico: `concrete` \| `nb` \| `tc` | — | O(1) |
| `engine_of()` | motor dueño del objeto (`nullptr` en constantes) | — | O(1) |
| `re_raw()` / `im_raw()` | valor crudo; solo constantes (si no, `logic_error`) | — | O(1) |
| `re_pattern()` / `im_pattern()` | patrón almacenado (dígitos NB o patrón TC), sin circuitos | — | O(1) |
| `real_rail()` / `imag_rail()` | riel NB canónico: constante → literales; `nb` → directo; `tc` → `tc_to_nb` | `detail::tc_to_nb` (§7.3) | O(W²) solo si `tc` |
| `tc_real_rail()` / `tc_imag_rail()` | patrón TC del datapath: constante → literales; `nb` → `nb_to_tc`; `tc` → directo | `detail::nb_to_tc` (§7.3) | O(W) solo si `nb` |
| `value(model)` | decodifica NB (`Σ d_i·(−2)^i`) o patrón TC con `wrap_nb`, y divide por `2^F` | `fixed_point::from_raw` | O(W) host, sin cláusulas |

**Categoría D — utilidades de host (sin circuito):** `fixed_point::trunc_shr(_i128)` y
`round_scaled_div` (§8), `core::is_constant` y `gates::all_constant` son predicados y
plegados O(W)/O(1) que implementan el plegado de constantes (ADR-010) en todas las capas.

Notas:

- `neg`/`conj`/`real`/`imag` sobre operandos simbólicos producen resultados de patrón
  TC (ADR-011), igual que `mul` (§10.3); sobre constantes producen una constante NB
  re-codificada (con `wrap_nb` en negativos — ADR-011).
- `eq_lit` con ambos operandos constantes no crea variables y devuelve el literal
  constante correcto; es el único camino de igualdad sobre circuitos y es lo que usa la
  restricción `mul(r, b) == a` de `operator/` (§10.4).

### 10.7 Operaciones derivadas — Etapa 2 (entregada)

Además de las operaciones de la Etapa 1, el núcleo entrega las operaciones derivadas de
la Etapa 2 (roadmap §15), con la misma dualidad concreta/simbólica y las mismas
políticas de escala (ADR-008/ADR-013) y plegado (ADR-010):

| Operación | Ruta concreta | Ruta simbólica (circuito) | Coste |
|---|---|---|---|
| `abs_sq(z)` | `z · conj(z)` por la ruta concreta de `mul`; `im_raw == 0` exacto | `z · conj(z)` por la ruta simbólica de `mul` (la parte imaginaria se anula estructuralmente: `ar·bi + ai·br = 0`); el desborde sigue la política de `mul` (§10.3) | 1× mul, O(W²) |
| `lt_lit(engine&, a, b)` | pliega a `true_lit`/`false_lit` comparando `(re, im)` lexicográficamente con alineación de escala (`__int128`) | `re_lt = ¬sle(b_re, a_re)`; `lt = re_lt ∨ (eq(re) ∧ ¬sle(b_im, a_im))` sobre rieles exactos alineados (ADR-013) | 1× EQ + 2× SLE + 1× AND + 1× OR, O(W) |
| `le_lit(engine&, a, b)` | ídem con `≤` | `re_lt ∨ (eq(re) ∧ sle(a_im, b_im))` | ídem |
| `pow(z, n)` (n entero) | square & multiply sobre la ruta concreta de `mul` (truncado por paso, ADR-003) | square & multiply sobre `operator*`; `n == 0` → `one`, `n == 1` → `z`; `n < 0` → `pow(one / z, −(n+1)) / z` (la división impone divisor no nulo, §10.4; `−(n+1)` evita el desborde de `INT_MIN`) | O(log n)× mul |
| `root_cbe(z, n)` | `z^(1/n)` con precisión de host y redondeo half-away-from-zero (ADR-003); fuera del rango NB → `std::out_of_range` | `y` libre con `y^n == z` (`eq_lit` añadido como cláusula unitaria, igual que `operator/`) | (n−1)× mul + 1× EQ |

Notas:

- La comparación lexicográfica es por tuplas `(re, im)` (orden del diccionario):
  `a < b ⟺ re_a < re_b ∨ (re_a == re_b ∧ im_a < im_b)`, y análogo para `≤`.
  `gt`/`ge` se obtienen invirtiendo los operandos: `lt_lit(e, b, a)` / `le_lit(e, b, a)`.
- `lt_lit`/`le_lit` devuelven un literal (como `eq_lit`): la imposición es explícita
  con `engine::add_unit`. El mismo objeto consigo mismo pliega: `lt_lit(e, z, z) ≡
  false_lit` y `le_lit(e, z, z) ≡ true_lit`.
- `root_cbe` exige `n >= 1` (`std::domain_error` en otro caso); `n == 1` devuelve `z`.
- `abs_sq(z)` equivale a `z · conj(z) == (|z|², 0)`; el truncado de `mul` se aplica
  al escalado `2^F` del módulo cuadrado (ADR-003/ADR-012).
- Estas operaciones reutilizan exclusivamente los circuitos de la Etapa 1; no
  introducen gates nuevos. `pow`/`abs_sq` son `constexpr` en la ruta concreta.

## 11. Solver: Kerberos (`satx::solver`)

La librería resuelve con el **kernel CDCL SLIME de Kerberos** (`src/kerberos/slime.c`,
CDCL industrial: propagación de unidades, dos literales vigilados, reinicios por EMA,
VSIDS/CHB, HESS, covertrace), integrado por su **C API embebida**
(`slime_sat_handle_create/solve/destroy`) sin ficheros intermedios. Todos los módulos C
de Kerberos (SLIME, BASILISK, PIXIE, WMIBO, GRINDER y aceleración) se compilan una sola
vez en la biblioteca `satx_kerberos`, que comparten el puente `satx::solver` y el
despachador CLI. El despachador `kerberos.c` (SLIME/BASILISK/PIXIE/WMIBO) sigue
disponible como herramienta CLI; la Etapa 5 conectará el conteo de modelos
(`dpll_wmc` en la referencia) al backend BASILISK.

```cpp
namespace satx::solver {
  enum class result { sat, unsat };

  struct options {                    // espejo de SlimeSatOptions (defaults de slime.c)
    int heuristic_mode = 0;           // 0 = VSIDS, 1 = CHB
    int use_mab = 0;  double mabc = 4.0;
    int use_hess = 0;
    int use_ct = 1;
    int ct_lbd_max = 6;  int ct_maxlen = 12;  int ct_max_cubes = 40000;
    int ct_buddy_merge = 0;  int ct_escape_rounds = 4;  int ct_probe_restarts = 4;
  };
  struct stats {                      // espejo de SlimeSatStats
    long long clauses, learnt;        // instantáneas absolutas
    long long conflicts, decisions, propagations, restarts;   // deltas por llamada
    long long hess_calls, hess_sat_hits, ct_added, ct_merged, ct_escaped, ct_probe_added;
  };

  class model {                                  // asignación de variables → bool
  public:
    bool get_var(std::int32_t var) const noexcept;  // 1-indexado por variable
    bool get(core::lit_t l) const noexcept;         // respeta el signo del literal
  };

  std::expected<model, result> solve(engine const&, std::size_t budget = 0);
  std::expected<model, result> solve(engine const&, options const&, stats* out = nullptr);
  std::expected<model, result> solve(engine const&, std::span<core::lit_t const> assumptions,
                                     options const& = {}, stats* out = nullptr);

  class session {                                  // handle CDCL reutilizado (RAII)
  public:
    explicit session(engine const&, options const& = {});
    std::expected<model, result> solve(std::span<core::lit_t const> assumptions = {},
                                       stats* out = nullptr);
    std::size_t variable_count() const noexcept;
    ~session();
  };
}
```

Nota de API: como `lit_t ≡ std::int32_t`, el acceso por variable usa `get_var` (no hay
sobrecarga posible entre ambos en C++).

- `solve` convierte la arena CNF del `engine` a los arrays de la C API (`nvars`,
  `clauses`, `sizes`), llama a `slime_sat_handle_solve` y traduce `rc`: 10 → `model`
  (vector `model01`), 20 → `result::unsat`, 0 → `std::runtime_error`.
- **Opciones y estadísticas (ADR-011):** `options` es el espejo de `SlimeSatOptions`
  (los defaults coinciden con `slime_sat_options_default`); `stats` de
  `SlimeSatStats` (clauses/learnt son instantáneas; el resto, deltas por llamada).
- **Suposiciones (ADR-011):** SAT bajo suposiciones (`assumptions`), con validación
  previa (literales ±v, 1 ≤ v ≤ nvars; 0/`INT32_MIN`/fuera de rango →
  `std::invalid_argument`). Útil para consultas incrementales y enumeración.
- **Sesiones (ADR-011):** `session` crea el handle una vez y reutiliza la base de
  cláusulas (incluidas las aprendidas) entre llamadas; las suposiciones se revierten
  tras cada `solve`. RAII destruye el handle.
- `budget` es **advisory** (ADR-009): el puente embebido de SLIME no impone límite de
  conflictos/nodos en esta etapa; el parámetro queda reservado para el futuro
  (external-stop con `SATX_HAVE_THREADS`).
- `engine::unsat()` (cláusula vacía o `[¬1]`) cortocircuita la llamada al kernel.
- La capa numérica no depende de `solver`: los circuitos son independientes del método
  de resolución; el modelo se inyecta en `value(model)`.

## 12. API C++26 — boceto esencial

```cpp
// consumo: #include <satx/satx.hpp>   (fachada de headers; no hay módulos C++26 aún)
namespace satx {

  using core::lit_t;

  class engine { /* §5.2 */ };

  // gates
  namespace gates {
    lit_t or2(engine&, lit_t, lit_t);
    lit_t and2(engine&, lit_t, lit_t);
    lit_t xor2(engine&, lit_t, lit_t);
    lit_t mux2(engine&, lit_t, lit_t, lit_t);
    lit_t fas(engine&, lit_t, lit_t, lit_t);
    lit_t fac(engine&, lit_t, lit_t, lit_t);
  }

  // números
  template<std::size_t W, std::size_t F> class complex;   // §9-10

  // Implementación: plantillas heterogéneas; W/F mixtos promueven a
  // W = max(Wa,Wb), F = max(Fa,Fb) con alineación de escala (ADR-008/ADR-013, §10).
  template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
  complex<std::max(Wa,Wb), std::max(Fa,Fb)> operator+(complex<Wa,Fa> const&,
                                                      complex<Wb,Fb> const&);

  // utilidades
  template<std::size_t W, std::size_t F> complex<W,F> conj(complex<W,F> const&);
  template<std::size_t W, std::size_t F> complex<W,F> neg (complex<W,F> const&);
  template<std::size_t W, std::size_t F> complex<W,F> real(complex<W,F> const&);
  template<std::size_t W, std::size_t F> complex<W,F> imag(complex<W,F> const&);
  template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
  core::lit_t eq_lit(engine&, complex<Wa,Fa> const&, complex<Wb,Fb> const&);

  // Etapa 2 (§10.7)
  template<std::size_t W, std::size_t F> constexpr complex<W,F> abs_sq(complex<W,F> const&);
  template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
  core::lit_t lt_lit(engine&, complex<Wa,Fa> const&, complex<Wb,Fb> const&);
  template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
  core::lit_t le_lit(engine&, complex<Wa,Fa> const&, complex<Wb,Fb> const&);
  template<std::size_t W, std::size_t F> constexpr complex<W,F> pow(complex<W,F> const&, int n);
  template<std::size_t W, std::size_t F> complex<W,F> root_cbe(complex<W,F> const&, int n);

  // igualdad de valor: solo constantes (sobre circuitos lanza std::logic_error;
  // la igualdad simbólica es eq_lit — §10.6)
  template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
  constexpr bool operator==(complex<Wa,Fa> const&, complex<Wb,Fb> const&);
  template<std::size_t Wa, std::size_t Fa, std::size_t Wb, std::size_t Fb>
  constexpr bool operator!=(complex<Wa,Fa> const&, complex<Wb,Fb> const&);
}
```

Uso esperado (ruta simbólica):

```cpp
satx::engine e;
satx::complex<16,4> x{e};                  // variable libre
satx::complex<16,4> y{1.5, -2.0};          // constante NB
auto z = (x + y) * x / satx::complex<16,4>::i_unit(e);
if (auto m = satx::solver::solve(e); m) {
    auto value = z.value(*m);              // decodificación desde el modelo de Kerberos
}
```

## 13. Estructura de directorios

```
satx/
├── CMakeLists.txt                 # C++26 + C17; Kerberos (kernel + backends) es
│                                  #   parte del producto en src/kerberos/
├── include/satx/
│   ├── satx.hpp                   # fachada única
│   ├── core/  (lit.hpp, clause.hpp, cnf.hpp, engine.hpp)
│   ├── gates/ (primitive.hpp, bitvec.hpp)
│   ├── num/   (negabinary.hpp, fixed_point.hpp, complex.hpp)
│   ├── quantum/ (gates.hpp, state.hpp, circuit.hpp, quantum.hpp)
│   ├── solver/(kerberos.hpp, model.hpp)
│   └── api.hpp                    # operadores y fábricas públicas re-exportadas
├── src/
│   ├── kerberos/                  # fuentes C de Kerberos (íntegras, propias):
│   │   │                          #   slime.c (CDCL), basilisk.c (#SAT), pixie.c,
│   │   │                          #   wmibo.c, grinder.c, kerberos.c (despachador),
│   │   │                          #   krb_parallel_*.c, krb_accel*.c/.cu,
│   │   │                          #   slime_bridge.h (C API embebida de SLIME)
│   └── solver/kerberos.cpp        # puente: arena CNF → slime_sat_handle_*
├── examples/                      # programas de demostración
│   ├── send_more_money.cpp        # criptoaritmética
│   ├── map_coloring.cpp           # coloreo de mapas
│   ├── nqueens.cpp                # N-reinas
│   ├── optimize_sum.cpp           # optimización por búsqueda binaria
│   ├── job_shop.cpp               # scheduling job-shop
│   ├── dice_distribution.cpp      # distribución de dados
│   ├── projectile.cpp             # tiro parabólico
│   ├── quantum_bell.cpp           # estados de Bell
│   ├── quantum_learning.cpp       # aprendizaje de compuertas (problema inverso)
│   ├── gaussian_integer_factorization.cpp  # factorización en Z[i] por SAT
│   ├── mandelbrot_escape.cpp      # Mandelbrot: c con tiempo de escape exacto
│   ├── quantum_teleportation.cpp  # teletransporte cuántico
│   ├── sudoku.cpp                 # Sudoku 9×9
│   ├── model_counting.cpp         # conteo de modelos por bloqueo + sesiones
│   └── complex_polynomial_roots.cpp  # raíces de p(z) por SAT
├── tests/                         # asserts habilitados SIEMPRE (-UNDEBUG)
│   ├── test_negabinary.cpp        # roundtrip exhaustivo del rango NB + oráculos
│   ├── test_complex_ops.cpp       # identidades y propiedades (ruta concreta)
│   ├── test_kerberos.cpp          # ruta simbólica resuelta por el kernel SLIME
│   ├── test_quantum_echoes.cpp    # OTOCs, eco, aprendizaje de B
│   └── test_solver_api.cpp        # opciones, stats, suposiciones, sesiones
├── 101.cpp                        # ejemplo mínimo (x + y == z)
├── docs/
│   ├── architecture.md            # este documento (fuente normativa)
│   └── manual.md                  # manual de usuario (API, recetas, excepciones)
├── README.md                      # construcción, prueba y panorámica del proyecto
└── LICENSE.txt                    # Apache-2.0
```

Kerberos es **parte integral del producto** — satx y Kerberos son un solo producto,
sin directorios `third_party/` ni variables de configuración externas: todos los módulos
C (`slime.c`, `basilisk.c`, `pixie.c`, `wmibo.c`, `grinder.c`, `krb_accel*.c`,
`krb_parallel_stub.c`) se compilan **una sola vez** en la biblioteca estática
`satx_kerberos` (con `SLIME_NO_MAIN`/`BASILISK_NO_MAIN`/`PIXIE_NO_MAIN`/`WMIBO_NO_MAIN`/
`GRINDER_NO_MAIN` y sin `SATX_HAVE_THREADS` — ruta serial), que comparten:
- el puente C++ `satx::solver` (`satx_solver`, `src/solver/kerberos.cpp`), que la enlaza
  vía la C API declarada en `src/kerberos/slime_bridge.h`; y
- el **despachador CLI `kerberos`** (`kerberos.c`, que solo aporta `main()`), construido
  como ejecutable con la opción `SATX_BUILD_KERBEROS_CLI` (ON por defecto).

## 14. Estrategia de pruebas

**Infraestructura (ADR-011):** la suite usa `assert` + bucles de propiedad; los targets
de test compilan con `-UNDEBUG` (o `/UNDEBUG` en MSVC) de modo que las aserciones se
ejecutan **siempre**, incluso en builds Release. (Antes de la revisión, NDEBUG dejaba
toda la suite inerte.) La suite cuenta con 5 tests (incluido `test_solver_api`).

1. **Roundtrip NB:** `decode(encode(n)) == n` para **todo** el rango `[min_NB(W), max_NB(W)]`
   con `W = 4, 8, 12, 16` (exhaustivo) más fronteras `constexpr` de W = 32.
2. **Conversiones NB↔TC:** `tc_to_nb(nb_to_tc(d)) == d` sobre el rango NB; para todo el
   rango TC, la salida es el representante canónico de la caja (`wrap_nb` del patrón) y
   la exactitud ⟺ el valor está en la caja NB (oráculo corregido, ADR-006 revisado;
   W = 4, 5, 6, 8, 12).
3. **Identidades complejas** (tolerancia `2^(−F)`): `i·i == −1`; `z + 0 == z`;
   `z·1 == z`; `(a+b)+c == a+(b+c)`; `a·(b+c) == a·b + a·c`; `z·conj(z) == |z|²`;
   `neg(z) + z == 0`; `conj(conj(z)) == z`; `eq_lit(e, z, z)` ≡ `true_lit` (también
   para `z` simbólico) y, sobre constantes, coincide con `operator==`.
4. **División:** `(a·b)/b == a` para `b ≠ 0`; `1/i == −i`; divisor de escala mixta.
5. **Ruta simbólica:** mismos tests contra el modelo del kernel SLIME de Kerberos
   (cruce de capas: `test_kerberos.cpp`), incluyendo la conversión NB→TC por circuito
   contra el oracle aritmético.
6. **Overflow:** `mul` con productos fuera de la caja NB → UNSAT (simbólico) / excepción
   (concreto); constantes fuera del rango NB → error.
7. **Etapa 2 (§10.7):** `abs_sq(z)` con `im == 0` exacto y `≈ |z|²`; `lt_lit`/`le_lit`
   lexicográficos sobre constantes (pliegue) y sobre circuitos (SAT y UNSAT);
   `pow(z, n)` contra el oracle del host y `pow(z, 4) == pow(z, 2)²`; `root_cbe(z, 2)`
   con `y² ≈ z` y `n < 1` → `std::domain_error`; todo también en la ruta simbólica.
8. **Regresiones de la revisión ADR-011/012/013:** identidad `x + 0 == x` en W impar
   (la caja NB ⊄ rango con signo de W bits); wrap de add coherente entre rutas
   (`85 + 85 == −86` en W=8); `eq_lit` con alineación exacta (antes vacuo cuando
   `F−Fa` desplazaba bits); `mul` truncado hacia cero (`(−0.5)·(0.0625) == 0` en
   ambas rutas); división con divisor de escala mixta (antes UNSAT espurio);
   `nb_exact_lit` barrido de patrones; guardas de `round_scaled_div` y `to_raw`;
   `model::get` con literales inválidos; unidad `[¬1]` → UNSAT; gates nuevos
   (`shr`/`sra`/bit a bit/`slt`/`ult`/`rca_carry`/`reduce_and`) contra el host;
   `width_policy::truncate`.
9. **Solver (ADR-011):** `test_solver_api` — suposiciones (SAT/UNSAT/validación),
   opciones + estadísticas, sesiones incrementales y conteo de modelos por cláusulas
   de bloqueo (`(a∨b)` → 3; `|x|² == 1` en CBE(4,0) → 4).

## 15. Roadmap de extensión (compatibilidad garantizada por esta arquitectura)

| Etapa | Contenido | Apoyo en este documento |
|---|---|---|
| 2 | ~~`abs_sq`, comparación lexicográfica; `pow` con exponente entero (square & multiply sobre circuitos); `root_cbe`~~ — **entregada** (§10.7) | rieles y `mul` |
| 2.5 | ~~Revisión ADR-011: canonicalización del datapath, `real`/`imag`, gates nuevos (`shr`/`sra`/bit a bit/`slt`/`ult`), opciones/stats/suposiciones/sesiones del solver~~ — **entregada** (§§6.1, 10, 11) | núcleo |
| 3 | **CORDIC** circular (sin/cos/tan) e hiperbólico (exp/log/sinh/cosh): ganancias `Kc ≈ 0.6072529350088814`, `Kh(N)` con índices repetidos `{4, 13, 40, 121}`; tablas `atan(2^−i)`, `atanh(2^−i)` como `constexpr` arrays; `pow_cbe`; divisor iterativo que sustituye la ruta simbólica exacta-en-cuadrícula de `operator/` | ruta simbólica + `engine` |
| 4 | `sqrt`, series (Taylor/Padé), factorial/gamma, más precisión adaptativa | `fixed_point` |
| 5 | `#SAT` (conteo de modelos vía backend BASILISK de Kerberos), monedas/pesos, ISA CCMASM (`C0..C31`) y kernel fetch/decode/execute/writeback con W/F dinámicos | `solver` + `complex` dinámico |
| 6 | Logos/QBF y capas superiores equivalentes (bajo demanda) | núcleo |

## 16. Apéndice — decisiones registradas (ADR)

- **ADR-001** — `W`/`F` como parámetros de plantilla; envoltorio dinámico en Etapa 5.
- **ADR-002** — Representación canónica **NB**; datapath sobre **patrones TC**;
  conversiones en la frontera. (Revisado por ADR-011.)
- **ADR-003** — Redondeo half-away-from-zero en constantes; truncado hacia cero en `mul`.
- **ADR-004** — `engine` explícito (sin estado global); literal 1 = VERDADERO.
- **ADR-005** — **Solver: Kerberos.** Los módulos C de Kerberos en `src/kerberos/` se
  compilan una sola vez en la biblioteca estática `satx_kerberos` (kernel y backends, con
  sus `*_NO_MAIN`) y `satx::solver::solve` invoca el kernel CDCL SLIME por su C API
  (`slime_sat_handle_*`, declarada en `src/kerberos/slime_bridge.h`), sin ficheros
  intermedios. Kerberos y satx son un solo producto; no se mantiene un DPLL propio.
- **ADR-006** — Recurrencia TC→NB **con resto completo** (§7.3). Revisado: la forma
  anterior con acarreo de un bit y condición «exacto ⟺ c_final == t[W−1]» era
  incorrecta; la exactitud ahora es la pertenencia a la caja NB (comparadores).
- **ADR-007** — La cláusula unitaria del `engine` es `[true_lit]` (`[+1]`, variable 1 =
  VERDADERO). La redacción original `[-true_lit]` procedía de la convención invertida
  `true`/`false` del ALU Python; la especificación C++ la corrige.
- **ADR-008** — Alineación de escala con F mixtos (la referencia Python no la hace):
  add/sub desplazan `F−Fa` bits (wrap); mul recorta con `K = min(Fa,Fb)`; div escala
  con `K = F+Fb−Fa` (§10). Las comparaciones usan alineación exacta (ADR-013).
- **ADR-009** — `solve(engine&, budget)`: `budget` es advisory; el puente embebido de
  SLIME no impone límite de conflictos en esta etapa (reservado para external-stop).
- **ADR-010** — Plegado de constantes en `gates`: los operandos `true_lit`/`false_lit` no
  consumen variables; rieles totalmente constantes se pliegan por aritmética directa
  (RCA/RCS/PM/eq). Los patrones TC de las constantes se materializan como literales
  (`v mod 2^W`), nunca por circuito.
- **ADR-011** — **Canonicalización del datapath.** Todo resultado simbólico se almacena
  como patrón TC de W bits decodificado con `wrap_nb` (representante de la caja NB de su
  clase mod 2^W). add/sub/neg/conj envuelven mod 2^W en **ambas** rutas; mul/div
  restringen el resultado a la caja NB (comparadores en W+1 bits) en **ambas** rutas.
  Entradas exactas del mul en W+1 bits (antes, valores de la caja NB se corrompían por
  sext: p. ej. `x + 0 ≠ x` para `x = 4` en W=3). Eliminado `const_policy` (muerto);
  `width_policy` implementado; guardas de `fixed_point` y de literales inválidos en
  `cnf`/`model`; `model::get(0) == false`; tests con `-UNDEBUG`.
- **ADR-012** — **Recorte hacia cero en `mul`.** La ventana `[K, K+W)` a secas es floor;
  la corrección `P + (signo ? 2^K−1 : 0)` antes de recortar produce el truncado
  simétrico de ADR-003 en la ruta simbólica, igualando la concreta.
- **ADR-013** — **Alineación exacta en comparaciones.** `eq_lit`/`lt_lit`/`le_lit`
  alinean escalas en el ancho exacto `max(Wa+F−Fa, Wb+F−Fb)+1` (sin wrap); con wrap,
  restricciones como `x == b` se volvían vacuas cuando el desplazamiento perdía bits.
- **ADR-014** — **Divisor no nulo con rieles nativos** (sin alineación de escala) en
  `operator/` simbólico; con rieles re-escalados, divisores válidos producían UNSAT
  espurio. Pliegues de `rca`/`rca_carry` con `__int128` (sin UB para anchos > 64).
- **ADR-015** — **Extensión del puente Kerberos:** `solver::options`/`solver::stats`
  (espejos de `SlimeSatOptions`/`SlimeSatStats`), `solve` con suposiciones y
  `solver::session` (handle CDCL reutilizado). Los ejemplos avanzados documentan el
  uso (enumeración por cláusulas de bloqueo, `mandelbrot_escape` con CHB).

