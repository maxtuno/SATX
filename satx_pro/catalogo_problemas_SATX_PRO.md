# Catálogo de problemas resolubles con SATX

**Oscar Riveros — _Obras Completas_, Vol. I — 2026-08-25**

Este catálogo describe una selección de problemas reales — con énfasis en
**logística y distribución, última milla, industria, mezclas y problemas
atípicos** — que pueden modelarse y resolverse con SATX, con
especificaciones formales, detalles de codificación y guías de escala.
Cada ficha indica el kernel recomendado:

| Kernel | Clase de problema | Salida |
|---|---|---|
| **SLIME** (CDCL) | Decisión SAT / UNSAT, testigo | modelo booleano |
| **BASILISK** | Conteo exacto de modelos (#SAT) | número exacto de soluciones |
| **PIXIE** | Programación lineal (LP) y mixta (MIP) | primal/dual continuo |
| **WMIBO** | Híbrido booleano-lineal con restricciones duras y blandas | óptimo + penalizaciones |

---

## 0. Guía rápida de selección de kernel

- **¿Solo importa si existe una solución?** → SLIME (decisión), con
  `--symmetry` activo por defecto para romper simetrías de forma sonora.
- **¿Cuántas soluciones hay exactamente?** → BASILISK (`count`,
  `count_projected` para marginales).
- **¿Todo es continuo y lineal?** → PIXIE (LP).
- **¿Hay enteros o binarios con costo/objetivo?** → PIXIE (MIP) o WMIBO si
  además hay restricciones blandas ponderadas.
- **¿El problema es booleano pero con preferencias ponderadas?** → WMIBO
  (cláusulas suaves + duras).
- **¿La estructura es no lineal (calidad de mezcla, trigonometría)?** →
  discretizar con la aritmética de punto fijo de SATX (CBE) y resolver por
  SAT, o linealizar por tramos en WMIBO.

> Regla de oro: todos estos problemas son NP-completos en su versión de
> optimización. Trabajar con instancias acotadas (ventanas temporales,
> horizonte corto, granularidad gruesa) y crecer iterativamente; usar
> `use_symmetry = 1`, suposiciones para fijar decisiones de negocio y
> sesiones para refinar el mismo modelo.

---

## 1. Logística y distribución

### 1.1 Asignación de almacenes a clientes (Facility Location)

- **Kernel**: PIXIE (MIP). **Tipo**: optimización de costo.
- **Descripción**: decidir qué almacenes abrir y qué clientes atiende cada
  uno, minimizando apertura + transporte.
- **Especificación**:
  - Datos: almacenes $W$, clientes $C$, costo fijo $f_w$, costo de servicio
    $c_{w,c}$, demanda $d_c$, capacidad $Q_w$.
  - Variables: $y_w\in\{0,1\}$ (abrir $w$), $x_{w,c}\in\{0,1\}$ (servir $c$
    desde $w$), o continuas si se permite fraccionar.
  - Restricciones:
    - cada cliente: $\sum_w x_{w,c} = 1$;
    - capacidad: $\sum_c d_c\,x_{w,c} \le Q_w\,y_w$;
    - acoplamiento: $x_{w,c}\le y_w$.
  - Objetivo: $\min \sum_w f_w y_w + \sum_{w,c} c_{w,c} d_c x_{w,c}$.
- **Codificación SATX**:
```cpp
#include <satx/satx.hpp>
satx::solver::pixie::model m;
for (auto& w : warehouses) m.add_binary("y_" + w);            // 1 = abrir
for (auto& [w, c] : pairs) m.add_binary("x_" + w + "_" + c);  // servir c desde w
for (auto& c : clients) {
    satx::solver::pixie::expr e;
    for (auto& w : warehouses) e += m.variable_at(idx(w, c)); // x_{w,c}
    m.add_constraint(e, satx::solver::pixie::compare::eq, 1.0);
}
// capacidad: Σ_c d_c·x_{w,c} ≤ Q_w·y_w ; objetivo: min Σ f_w·y_w + Σ c·d·x
m.set_objective(/*expr de costo*/, satx::solver::pixie::sense::min);
auto sol = m.solve();   // valores por variable (sol.*(v)) y sol.objective()
```
- **Escala**: cientos de almacenes × miles de clientes es rutinario en LP
  relajado; la versión entera con 50–200 binarias resuelve en segundos.
- **Variantes**: multi-producto, almacenes por niveles (central–regional),
  incertidumbre (escenarios con suposiciones).

### 1.2 Rutas de vehículos con capacidad (CVRP, decisión y óptimo)

- **Kernel**: WMIBO (MIP híbrido) para el óptimo; SLIME para factibilidad
  con horizonte fijo.
- **Especificación**:
  - Datos: depósito 0, clientes $1..n$ con demanda $d_i$, flota $K$ con
    capacidad $Q$, distancias $t_{i,j}$.
  - Variables: $x_{i,j,k}\in\{0,1\}$ (el vehículo $k$ viaja $i\to j$);
    $u_i\in[0,n]$ (orden de visita, MTZ).
  - Restricciones:
    - grado: $\sum_j x_{i,j,k}=\sum_j x_{j,i,k}=$ 1 si $i$ cliente, $k$ único;
    - eliminar subciclos (MTZ): $u_i-u_j+1\le n(1-x_{i,j,k})$;
    - capacidad: $\sum_{i\ne0} d_i\sum_j x_{i,j,k}\le Q$;
    - cada cliente atendido por un vehículo: $\sum_{k,j}x_{i,j,k}=1$.
  - Objetivo: $\min\sum_{k,i,j}t_{i,j}x_{i,j,k}$.
- **Codificación SATX** (WMIBO):
```cpp
satx::solver::wmibo::model m;
for (int k = 0; k < K; ++k)
    for (auto& [i, j] : arcs)
        m.add_boolean("x_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k));
// capacidad por vehículo (restricción dura), usando referencias guardadas:
for (int k = 0; k < K; ++k) {
    satx::solver::wmibo::expr cap;
    for (auto& [i, j] : arcs) if (i != 0) cap += d[i] * m.variable_at(idx(i, j, k));
    m.add_constraint(cap, satx::solver::wmibo::compare::le, Q);
}
m.set_objective(/* Σ t_ij · x_ijk */, satx::solver::wmibo::sense::min);
```
- **Escala**: hasta ~50 clientes con MTZ; para más, codificación SAT pura
  con tiempos de visita (Time-indexed) o particionar por clúster
  (cluster-first, route-second: resolver 1.1 y luego CVRP por clúster).
- **Variantes**: ventanas de tiempo (1.4), flota heterogénea, retornos
  (backhauls), pick-up & delivery.

### 1.3 Cubicaje: carga de contenedores 2D/3D (Bin Packing)

- **Kernel**: SLIME (decisión) con WMIBO si se optimiza volumen sobrante.
- **Especificación (2D guillotinable)**:
  - Datos: paletas $P$, cajas con dimensiones $(w_i,h_i)$, contenedor $W\times H$.
  - Variables: $x_{i,p}\in\{0,1\}$ (caja $i$ en paleta $p$); coordenadas
    discretizadas $cx_i, cy_i$ sobre la rejilla.
  - Restricciones:
    - cada caja, una paleta: AMO sobre $\{x_{i,p}\}_p$;
    - no solapamiento: para cada par $(i,j)$ en la misma paleta,
      $cx_i+w_i\le cx_j \lor cx_j+w_j\le cx_i \lor$ (análogo en $y$) —
      codificado con literales auxiliares;
    - límites: $cx_i + w_i \le W$, $cy_i + h_i \le H$ (con aritmética CBE o
      cotas pre-codificadas).
- **Codificación SATX (esqueleto)**:
```cpp
satx::engine e;
auto box_in = [&](int i, int p) { return e.add_variable(); };       // x_{i,p}
// exactamente-uno: cláusulas por pares (AMO) + al-menos-uno (ALO)
for (int i = 0; i < n; ++i) {
    for (int p1 = 0; p1 < P; ++p1)
        for (int p2 = p1 + 1; p2 < P; ++p2)
            e.add_clause({satx::core::neg(box_in(i, p1)), satx::core::neg(box_in(i, p2))});
    e.add_clause({/* ALO: todos los box_in(i,·) */});
}
// no solapamiento dentro de la paleta con variables de posición en rejilla...
auto sol = satx::solver::solve(e);
```
- **Escala**: decenas de cajas por contenedor; la rejilla domina el tamaño
  (granularidad 1 cm → miles de literales; granularidad 5–10 cm → segundos).
- **Variantes**: 3D, orientación/rotación permitida, fragilidad, peso por eje
  (centro de gravedad), carga mixta camión–contenedor.

### 1.4 Programación de muelles de carga (Dock Scheduling)

- **Kernel**: SLIME (time-indexed) o WMIBO.
- **Especificación**:
  - Datos: muelles $D$, camiones $T$, ventanas de llegada $[a_t,b_t]$,
    duración de descarga $p_t$, incompatibilidades (refrigerado, peligroso).
  - Variables: $x_{t,d,s}\in\{0,1\}$ (camión $t$ en muelle $d$ inicia en el
    instante $s$).
  - Restricciones:
    - cada camión, una vez: $\sum_{d,s}x_{t,d,s}=1$ (dentro de su ventana);
    - sin solapamiento por muelle: $\sum_t\sum_{s'\in(s-p_t,s]}x_{t,d,s'}\le1$
      para cada $(d,s)$ (cardinalidad ≤ 1);
    - incompatibilidad: $\neg x_{t,d,s}\lor\neg x_{t',d,s'}$ si $(t,t')$
      incompatibles y los intervalos se tocan.
  - Objetivo (WMIBO): minimizar $\sum s\cdot x_{t,d,s}$ o el makespan con
    una variable entera auxiliar $z\ge s+p_t$ para cada asignación activa.
- **Escala**: 10 muelles × 50 camiones × horizonte 24 h (granularidad 15 min)
  → ~10⁵ literales; resuelve en segundos con `--symmetry`.
- **Variantes**: prioridades de cliente, cross-docking (transbordo entre
  camiones de larga distancia y reparto), recursos compartidos (bandas,
  montacargas).

### 1.5 Cross-docking con transbordo

- **Kernel**: SLIME / WMIBO.
- **Descripción**: emparejar llegadas y salidas de camiones para minimizar
  el tiempo de permanencia y las cargas movidas, respetando capacidades de
  puerta y compatibilidades de producto.
- **Variables**: $y_{i,j}$ (la carga de llegada $i$ alimenta la salida $j$),
  $s_i$ (instante de inicio en puerta).
- **Restricciones**: balance de flujo por producto, capacidad de puerta
  (una operación a la vez), precedencia $s_i+p_i\le s_j$ cuando $y_{i,j}=1$
  y se exige secuencia (big-M en WMIBO).
- **Objetivo**: minimizar $\sum(s_i+p_i)$ (tiempo total) o el número de
  puertas usadas.
- **Variantes**: productos fríos (cadena de frío, precedencia estricta),
  consolidación multi-origen.

---

## 2. Última milla

### 2.1 Reparto con ventanas de tiempo y flota heterogénea

- **Kernel**: SLIME (decisión) / WMIBO (costo).
- **Especificación**:
  - Datos: pedidos con ventanas $[e_i,l_i]$, flota con capacidades y costos
    por km distintos, zonas restringidas.
  - Variables: $x_{i,j,k}$ (arco por vehículo), $t_i$ (instante de llegada).
  - Restricciones:
    - rutas por vehículo (grado) y capacidad;
    - ventanas: $e_i\le t_i\le l_i$; continuidad: $t_j\ge t_i+p_i+t_{i,j}-M(1-x_{i,j,k})$;
    - zona: $x_{i,j,k}=0$ si $k$ no opera la zona de $j$.
  - Objetivo: minimizar costo total (distancia + vehículos usados); en
    WMIBO, cláusulas blandas penalizan entregas fuera de ventana.
- **Escala**: 100–200 paradas por ruta en decisión pura; para el óptimo,
  resolver por zonas (1.1) y luego 2.1 por zona.
- **Variantes**: entregas fallidas y reintentos, depósitos móviles,
  ventanas acordadas con cliente (elección de franja → oferta comercial).

### 2.2 Agrupación de pedidos (Order Batching)

- **Kernel**: WMIBO.
- **Descripción**: agrupar pedidos de un almacén en oleadas de picking de
  modo que se minimice el recorrido total y se respeten los plazos.
- **Variables**: $y_{i,b}$ (pedido $i$ en oleada $b$); enteras $u_b$
  (inicio de oleada).
- **Restricciones**: cada pedido en una oleada; capacidad de carro;
  deadline: $u_b + T_b \le l_i$ para pedidos urgentes (big-M).
- **Objetivo**: minimizar $\sum_b f_b$ (costo de oleada, con cláusulas
  blandas por pedido tardío con peso por cliente).
- **Escala**: miles de pedidos × decenas de oleadas; las cláusulas blandas
  permiten infactibilidades parciales negociadas (el cliente paga la
  prioridad).

### 2.3 Asignación de casilleros y puntos de recogida

- **Kernel**: SLIME / WMIBO.
- **Descripción**: asignar pedidos a casilleros (lockers) o tiendas de
  recogida minimizando la distancia peatonal del cliente y las colisiones
  de uso simultáneo.
- **Variables**: $x_{p,l}$ (pedido $p$ al casillero $l$); $y_{p,t}$ (el
  pedido ocupa el casillero en el turno $t$).
- **Restricciones**: un casillero por pedido y turno (AMO), capacidad por
  casillero, incompatibilidad de productos (frío/seco), reservas
  preexistentes (suposiciones fijas).
- **Objetivo**: minimizar la distancia total o el número de casilleros
  usados; #SAT (BASILISK) cuenta las configuraciones admisibles para
  dimensionar la red.
- **Variantes**: ubicación óptima de casilleros nuevos (combinar con 1.1:
  candidatos como "almacenes" sin costo fijo de apertura).

### 2.4 Rutas de técnicos con habilidades (Workforce Scheduling)

- **Kernel**: SLIME (decisión) / WMIBO (costo y blandas).
- **Especificación**:
  - Datos: servicios con duración, ventana y habilidad requerida
    $h(s)\in H$; técnicos con habilidades, horarios y zonas.
  - Variables: $x_{t,s}$ (técnico $t$ atiende servicio $s$); $o_{s,s',t}$
    (orden de atención); $u_s$ (instante de inicio).
  - Restricciones: cobertura $\sum_t x_{t,s}=1$; habilidad $x_{t,s}=0$ si
    $h(s)\notin H_t$; sin solapamiento por técnico (intervalos disjuntos);
    ventanas; descansos y turnos legales (máximo de horas, pausa obligada).
- **Objetivo**: minimizar desplazamientos + horas extra; blandas: preferencia
  de técnico por zona, cliente preferente.
- **Escala**: 20 técnicos × 100 servicios por día en decisión; horizonte
  diario para acotar.
- **Variantes**: mantenimiento preventivo periódico (1.9 combinado),
  emergencias con prioridad (blandas de alto peso).

### 2.5 Reparto con drones (TSP-D)

- **Kernel**: WMIBO / SLIME.
- **Descripción**: un camión con $k$ drones realiza entregas; los drones
  despegan y aterrizan en el camión solo en paradas.
- **Variables**: $x_{i,j}$ (ruta del camión), $d_{l,i,j}$ (el dron $l$ va
  de la parada $i$ a la parada $j$ entregando el paquete $p$), $t_i$
  (tiempo de llegada del camión).
- **Restricciones**: cada paquete entregado por camión o dron; autonomía
  (batería → cota de distancia); sincronización: el dron sale y vuelve
  solo cuando el camión está en la parada ($t$-indexado).
- **Objetivo**: minimizar el makespan (tiempo total de ruta).
- **Escala**: 1 camión + 2–4 drones + 20–50 paquetes como referencia de
  complejidad; es fuertemente NP-completo — usar descomposición.

---

## 3. Industria

### 3.1 Programación de producción Job Shop / Flow Shop

- **Kernel**: SLIME (time-indexed, decisión de makespan ≤ B) / WMIBO.
- **Especificación**:
  - Datos: trabajos $J$, máquinas $M$, procesamientos $p_{j,m}$, secuencia
    tecnológica (orden de máquinas por trabajo).
  - Variables: $x_{j,m,t}$ (el trabajo $j$ se procesa en $m$ iniciando en
    $t$).
  - Restricciones:
    - cada operación se ejecuta una vez: $\sum_t x_{j,m,t}=1$;
    - capacidad por máquina: $\sum_j\sum_{t'\in(t-p_{j,m},t]}x_{j,m,t'}\le1$;
    - precedencia tecnológica: la operación siguiente no inicia antes de
      $t+p_{j,m}$;
    - makespan: $t+p_{j,m}\le B$ para todo $(j,m,t)$ activo.
  - Objetivo: minimizar $B$ (búsqueda binaria sobre $B$ con SLIME, o
    variable entera en WMIBO).
- **Escala**: 10 trabajos × 10 máquinas × horizonte 100 → ~10⁴ variables;
  con `--symmetry` se rompen máquinas idénticas.
- **Variantes**: tiempos de preparación dependientes de la secuencia
  (changeover), recursos renovables (operarios), paradas de máquina
  (mantenimiento planificado como trabajos bloqueantes).

### 3.2 Secuenciación de coladas y cambios de aleación (Lot-sizing + Changeover)

- **Kernel**: WMIBO.
- **Descripción**: planificar qué aleación se funde en cada horno y en qué
  orden, minimizando costos de cambio (limpieza de crisol) y atrasos.
- **Variables**: $y_{c,t}$ (aleación $c$ en el período $t$), $z_{c,c',t}$
  (cambio de $c$ a $c'$ en $t$), inventario entero $I_{c,t}$.
- **Restricciones**:
  - un producto por horno y período;
  - balance de inventario $I_{c,t}=I_{c,t-1}+q_{c,t}-d_{c,t}$;
  - lote mínimo (si se produce, $q_{c,t}\ge q^{\min}_c$);
  - cambio: $z_{c,c',t}\ge y_{c,t-1}+y_{c',t}-1$.
- **Objetivo**: minimizar $\sum c^{\text{cambio}}_{c,c'}z+\sum
  c^{\text{almacén}}_c I_{c,t}$ + blandas por atraso de pedido.
- **Escala**: 10 aleaciones × 30 períodos → MIP pequeño, segundos.
- **Variantes**: dos hornos paralelos (sincronización), tiempos de
  calentamiento dependientes del producto.

### 3.3 Corte de materiales (Cutting Stock / Guillotine)

- **Kernel**: PIXIE (LP columna) o SLIME (patrones exactos).
- **Descripción**: cortar rollos o planchas estándar en piezas demandadas
  minimizando el desperdicio.
- **Variables (enumeración de patrones)**: $x_p$ = número de planchas
  cortadas con el patrón $p$ (patrones generados con SLIME).
- **Restricciones**: $\sum_p a_{i,p}x_p \ge d_i$ (demanda por pieza);
  $x_p\in\mathbb{Z}_{\ge0}$.
- **Objetivo**: minimizar $\sum_p x_p$ (o el área sobrante).
- **Escala**: generación de patrones con SLIME (decisión "¿existe un patrón
  con estas piezas en una plancha?") + LP maestro con PIXIE; iteración de
  generación de columnas.
- **Variantes**: 2D no guillotinable (relacionado con 1.3), restos
  reaprovechables como inventario.

### 3.4 Planificación de mantenimiento preventivo

- **Kernel**: SLIME (time-indexed).
- **Descripción**: calendarizar tareas de mantenimiento con periodicidad
  máxima, precedencias y recursos (cuadrillas, repuestos).
- **Variables**: $x_{m,t}$ (mantenimiento de máquina $m$ inicia en $t$);
  $r_{k,t}$ (recurso $k$ ocupado en $t$).
- **Restricciones**:
  - periodicidad: entre dos tareas consecutivas de $m$, $\Delta\le P_m$
    (ventanas deslizantes: $\sum_{t'=t}^{t+P_m}x_{m,t'}\ge1$);
  - duración y recursos: $\sum_{m,t'\in(t-p_m,t]}x_{m,t'}\le R_k$;
  - no interrumpir producción crítica (bloqueos declarados).
- **Objetivo**: minimizar intervenciones (o maximizar disponibilidad con
  WMIBO y blandas).
- **Escala**: 50 máquinas × horizonte anual por semanas → factible; es el
  problema de mantenimiento con periodicidad (clásico).

### 3.5 Verificación de circuitos y generación de patrones de prueba

- **Kernel**: SLIME (nativo).
- **Descripción**: (a) verificar equivalencia de dos circuitos (miter:
  XOR de salidas → UNSAT ⟹ equivalentes); (b) generar un vector de prueba
  que distinga (SAT del miter); (c) atascos (stuck-at): distinguir el
  circuito sano del circuito con la falla.
- **Codificación**: las compuertas se codifican con cláusulas de Tseitin
  (la infraestructura `satx::engine` ya genera circuitos aritméticos con
  CBE: cada compuerta es un bloque de cláusulas).
- **Objetivo**: decisión SAT/UNSAT; #SAT (BASILISK) cuenta los vectores que
  detectan una falla (cobertura de test).
- **Escala**: circuitos con decenas de miles de compuertas; añadir
  `--symmetry` para circuitos con subbloques repetidos.

### 3.6 Paletización (patrones de apilado)

- **Kernel**: SLIME.
- **Descripción**: decidir el patrón de cajas por capa de paleta y el
  número de capas, con estabilidad y peso máximo.
- **Variables**: $x_{i,pos}$ (caja $i$ en posición de la rejilla de capa);
  $L_c$ (capa $c$ activa).
- **Restricciones**: rejilla con solapamiento prohibido (como 1.3), peso
  acumulado por columna ≤ máximo, capas contiguas (sin huecos intermedios
  para estabilidad).
- **Objetivo**: maximizar cajas por paleta (o minimizar paletas con WMIBO).
- **Variantes**: cajas de varios tamaños, capas alternadas (trabazón).

---

## 4. Mezclas (blending)

### 4.1 Dieta / mezcla de alimentos (LP clásico)

- **Kernel**: PIXIE (LP).
- **Especificación**:
  - Datos: ingredientes con costo $c_i$ y nutrientes $a_{n,i}$; mínimos y
    máximos nutricionales $[L_n,U_n]$.
  - Variables: $x_i\ge0$ (cantidad de ingrediente $i$).
  - Restricciones: $L_n\le\sum_i a_{n,i}x_i\le U_n$ para cada nutriente;
    proporciones ($x_i\le p\cdot\sum_j x_j$).
  - Objetivo: $\min\sum_i c_i x_i$.
- **Escala**: cientos de ingredientes × decenas de nutrientes: milisegundos.
- **Variantes**: enteros (paquetes discretos → MIP), sensibilidad (rangos de
  costos donde la solución no cambia — análisis dual).

### 4.2 Mezcla de minerales/carbón con calidad (MIP)

- **Kernel**: PIXIE (MIP) / WMIBO.
- **Especificación**:
  - Datos: pilas de mineral con calidades $q_{i,k}$ (ley, humedad, ceniza,
    azufre) y disponibilidad $A_i$; pedidos con bandas de calidad
    $[L_{p,k},U_{p,k}]$ y tonelaje $D_p$.
  - Variables: $x_{i,p}\ge0$ (toneladas de pila $i$ al pedido $p$);
    $y_{i,p}\in\{0,1\}$ si se exige lote mínimo.
  - Restricciones:
    - cumplimiento: $\sum_i x_{i,p}=D_p$;
    - disponibilidad: $\sum_p x_{i,p}\le A_i$;
    - calidad por pedido: $L_{p,k}D_p\le\sum_i q_{i,k}x_{i,p}\le U_{p,k}D_p$
      (¡la restricción es lineal en $x$!).
    - lote mínimo: $x_{i,p}\ge q^{\min}y_{i,p}$, $x_{i,p}\le A_i y_{i,p}$.
  - Objetivo: minimizar el costo de uso de material noble, o el tonelaje de
    la pila más barata; WMIBO: blandas para pedidos que toleran bandas.
- **Escala**: 100 pilas × 20 pedidos × 10 calidades → MIP mediano, segundos
  a minutos; los lotes mínimos binarios son la parte dura.
- **Variantes**: múltiples períodos (agotamiento de pilas), mezclas en
  línea (nivel de silo).

### 4.3 Mezcla de crudos con índices no lineales

- **Kernel**: WMIBO (linealizado por tramos) o SLIME (discretización
  aritmética exacta con CBE).
- **Descripción**: combinar crudos para cumplir especificaciones de refino
  (densidad, azufre, octanaje de cortes) cuando las propiedades no se
  mezclan linealmente (p. ej. índice de viscosidad, octanaje).
- **Especificación**:
  - Variables: $x_i$ (fracción de crudo $i$), con $\sum x_i=1$;
  - propiedad $k$: $P_k = g_k(x)$ con $g_k$ cóncava/convexa → aproximación
    por tramos: $P_k=\sum_\ell \lambda_\ell g_k(x_\ell)$ con SOS2.
  - Restricciones: bandas de producto $[L_k,U_k]$, disponibilidad y
    compatibilidad de crudos (corrosivos, ácidos).
- **Codificación SATX (discretización exacta)**: fijar un paso de mezcla
  (p. ej. 1 %) y modelar $\sum x_i=1$ sobre enteros con las restricciones
  aritméticas del `engine`; la no linealidad se evalúa tabla a tabla
  (`g_k` precalculada), generando una cláusula por celda inválida.
- **Escala**: 10 crudos × 5 propiedades × paso 1 % → ~10⁴ celdas; SAT decide
  en segundos; #SAT cuenta las recetas factibles (espacio de diseño).
- **Variantes**: multiperíodo con inventario de tanques, mezclas de gasolina
  con RVP/oxigenados.

### 4.4 Mezclas de productos químicos con pureza y reactividad

- **Kernel**: SLIME / WMIBO.
- **Descripción**: formular un lote con concentraciones objetivo e
  incompatibilidades binarias (reactivos que no pueden coexistir).
- **Variables**: $x_i$ continuas (o discretizadas) de cada componente;
  $y_i$ binarias (componente presente).
- **Restricciones**: balance de masa, pureza $\sum_i p_i x_i \ge P\cdot
  \sum_i x_i$, incompatibilidad $\neg y_i \lor \neg y_j$, estabilidad
  (vida útil conjunta ≥ mínimo).
- **Objetivo**: minimizar costo; blandas: preferencia por proveedor,
  restricción de stock.

---

## 5. Problemas atípicos (SATX como herramienta universal)

### 5.1 Programación de personal con reglas legales (Nurse Rostering)

- **Kernel**: SLIME.
- **Descripción**: asignar turnos (D/E/N, libre) a enfermeras por día
  cumpliendo cobertura, contratos y reglas.
- **Variables**: $x_{n,d,s}\in\{0,1\}$.
- **Restricciones**:
  - cobertura: $\sum_n x_{n,d,s}\ge \text{req}_{d,s}$ (cardinalidad);
  - exactamente un turno por enfermera y día;
  - no N seguido de D; máximo de noches consecutivas; descanso tras noche;
  - horas contractuales mensuales (sumas con CBE o cardinalidad).
- **Objetivo**: factibilidad; WMIBO con blandas (peticiones de día libre,
  equidad de fines de semana).
- **Escala**: 20 enfermeras × 28 días → ~10⁴ literales, segundos; es un
  benchmark clásico de SAT competitivo.

### 5.2 Calendarios deportivos (Round-Robin con restricciones)

- **Kernel**: SLIME / WMIBO.
- **Descripción**: programar una liga de $n$ equipos (impar → fantasma) con
  localía alternada, derbis no coincidentes, estadios compartidos, TV.
- **Variables**: $x_{i,j,r}\in\{0,1\}$ (en la ronda $r$, $i$ recibe a $j$);
  $h_{i,r}$ (localía de $i$).
- **Restricciones**: todos contra todos (exactamente una vez por par, en
  alguna ronda), un partido por equipo y ronda, límite de partidos de local
  consecutivos, pares que no coinciden en casa (estadio compartido).
- **Escala**: 20 equipos × 19 rondas → decenas de miles de variables; es el
  clásico de la literatura de "break minimization" (WMIBO: minimizar
  cambios de localía con blandas).

### 5.3 Diseño de experimentos: arrays de cobertura (Covering Arrays)

- **Kernel**: BASILISK (#SAT) + SLIME.
- **Descripción**: encontrar el mínimo conjunto de pruebas que cubre toda
  combinación de $t$ parámetros (fuerza $t$), clave en pruebas de software
  y de configuración industrial.
- **Variables**: $x_{r,p,v}$ (fila de prueba $r$, parámetro $p$, valor $v$).
- **Restricciones**:
  - un valor por parámetro y fila;
  - cobertura: para cada $t$-tupla $(p_1..p_t, v_1..v_t)$, al menos una
    fila la realiza: $\bigvee_r \bigwedge x_{r,p_i,v_i}$ (generar con
    BASILISK por enumeración o con SLIME incremental).
- **Objetivo**: minimizar filas (búsqueda binaria); #SAT cuenta diseños
  admisibles para evaluar margen.
- **Escala**: fuerza 2 con 10 parámetros × 5 valores → decenas de filas.

### 5.4 Generación procedural de mapas y niveles

- **Kernel**: SLIME.
- **Descripción**: generar mapas de videojuego que cumplan reglas: sala
  inicial y final conectadas, llaves antes que puertas, $k$ enemigos con
  distancia mínima, simetría estética.
- **Variables**: $c_{x,y}$ (celda = muro/suelo), $r_i$ (sala $i$), $d_e$
  (distancia entre entidades, con CBE).
- **Restricciones**: conectividad (flujo o alcanzabilidad por pasos),
  precedencia llave–puerta (camino sin la puerta), cotas de dificultad
  (número de celdas transitables).
- **Salida**: cada modelo es un mapa válido; re-ejecutar con suposiciones
  distintas (semilla) produce variantes; BASILISK cuenta cuántos mapas
  existen (dimensionamiento de contenido).
- **Escala**: mapas de 20×20 con presupuesto de distancia → segundos.

### 5.5 Composición musical con reglas

- **Kernel**: SLIME / BASILISK.
- **Descripción**: generar melodías y progresiones que cumplen reglas de
  contrapunto o estilo: notas en escala, resolución de sensibles,
  prohibición de quintas paralelas, forma (ABA).
- **Variables**: $n_{t,p}$ (nota $p$ en el pulso $t$); $ch_{c,t}$ (acorde
  $c$ en el compás $t$).
- **Restricciones**: exactamente una nota por pulso; consonancia con el
  acorde activo; movimientos prohibidos entre pulsos consecutivos
  (cláusulas $\neg n_{t,p}\lor\neg n_{t+1,q}$); repetición temática (ABA:
  $n_{t,p}=n_{T-t,p}$).
- **Salida**: BASILISK enumera las composiciones del estilo (espacio
  creativo medible); SLIME devuelve una por semilla.

### 5.6 Rompecabezas y juegos lógicos

- **Kernel**: SLIME / BASILISK.
- **Ejemplos**: Sudoku $n^2\times n^2$, Nonogramas, Slitherlink, rompecabezas
  tipo Einstein/Zebra, cripto-aritmética (SEND+MORE=MONEY con sumadores CBE).
- **Codificación**: variables $x_{i,j,d}$ (celda $(i,j)$ con dígito $d$);
  AMO/ALO por celda, fila, columna y caja; BASILISK determina si la
  solución es única (unicidad de la pista) contando modelos.
- **Escala**: Sudoku 9×9 → 729 variables, instantáneo; 25×25 → miles de
  variables, segundos con `--symmetry` (simetrías de dígitos rotas
  automáticamente por el pase de simetrías de SATX).

### 5.7 Factorización de enteros gaussianos y aritmética exótica

- **Kernel**: SLIME (ya en los ejemplos: `gaussian_integer_factorization`).
- **Descripción**: factorizar $z=a+bi$ en $\mathbb{Z}[i]$ modelando la
  multiplicación como circuito CBE (negabinario para signos sin sesgo) y
  resolviendo por SAT.
- **Variables**: bits de los factores; cláusulas del multiplicador.
- **Escala**: enteros de 16–32 bits por factor → segundos; la complejidad
  crece como el tamaño del circuito (no es criptográficamente competitivo,
  pero ilustra SAT sobre aritmética exacta).

### 5.8 Física inversa: aprendizaje de operadores cuánticos

- **Kernel**: SLIME (ejemplo `quantum_learning`).
- **Descripción**: dado un dato medido (OTOC) de un circuito cuántico,
  reconstruir la compuerta desconocida $B$ modelada con coeficientes
  simbólicos, unitaridad y realidad, de modo que el eco reproduzca el dato.
- **Variables**: bits de los coeficientes de $B$ (punto fijo W,F); la
  evolución del circuito se compila a restricciones SAT (CBE/CORDIC).
- **Salida**: la matriz $B$ aprendida verifica contra el oráculo numérico.

### 5.9 Planificación de movimientos de robot (discretizada)

- **Kernel**: SLIME.
- **Descripción**: mover un robot (o brazo) en una rejilla con obstáculos,
  evitando colisiones y minimizando pasos.
- **Variables**: $x_{t,celda}$ (el robot está en la celda en el paso $t$);
  AMO por paso; transición válida entre pasos (adyacencia libre);
  obstáculos fijos.
- **Objetivo**: factibilidad con horizonte acotado (búsqueda binaria del
  mínimo); multi-robot con colisión mutua (AMO sobre celdas por paso).
- **Variantes**: brazos articulados (configuraciones como celdas de un
  C-space discretizado), rutas con zonas de exclusión temporal.

### 5.10 Planificación de menús con nutrición y presupuesto

- **Kernel**: SLIME / WMIBO.
- **Descripción**: menú semanal (desayuno/almuerzo/cena) con variedad,
  presupuesto, alergias y balance nutricional (combinar con 4.1).
- **Variables**: $x_{d,m,p}$ (plato $p$ el día $d$ en la comida $m$).
- **Restricciones**: un plato por comida; no repetir el mismo plato más de
  $k$ veces por semana; alergias del comensal excluyen platos; presupuesto
  semanal (suma de costos); nutrientes agregados (ventanas).
- **Salida**: WMIBO pondera preferencias como blandas (platos favoritos).

### 5.11 Rutas con métricas de sostenibilidad (CO₂)

- **Kernel**: WMIBO.
- **Descripción**: plan de rutas (como 1.2/2.1) con objetivo multi-criterio:
  distancia, consumo (dependiente de carga → tramos con costo incremental),
  emisiones por zona (LEZ), ventanas.
- **Modelo**: cláusulas duras de factibilidad; blandas: $w_1$·km +
  $w_2$·CO₂ + $w_3$·retrasos; explorar el frente variando pesos
  (re-ejecuciones con suposiciones de presupuesto por métrica).
- **Salida**: planes Pareto-comparables para el decisor.

---

## 6. Tabla resumen

| # | Problema | Sección | Kernel típico | Tipo |
|---|---|---|---|---|
| 1 | Facility location | 1.1 | PIXIE MIP | opt |
| 2 | CVRP | 1.2 | WMIBO | opt |
| 3 | Cubicaje 2D/3D | 1.3 | SLIME | dec/opt |
| 4 | Muelles de carga | 1.4 | SLIME | dec/opt |
| 5 | Cross-docking | 1.5 | SLIME/WMIBO | dec/opt |
| 6 | Ventanas + flota | 2.1 | SLIME/WMIBO | dec/opt |
| 7 | Order batching | 2.2 | WMIBO | opt |
| 8 | Casilleros | 2.3 | SLIME/BASILISK | dec/count |
| 9 | Técnicos + habilidades | 2.4 | SLIME/WMIBO | dec/opt |
| 10 | TSP-D | 2.5 | WMIBO | opt |
| 11 | Job shop | 3.1 | SLIME/WMIBO | dec/opt |
| 12 | Coladas/changeover | 3.2 | WMIBO | opt |
| 13 | Cutting stock | 3.3 | PIXIE+SLIME | opt |
| 14 | Mantenimiento | 3.4 | SLIME | dec |
| 15 | Circuitos/pruebas | 3.5 | SLIME/BASILISK | dec/count |
| 16 | Paletización | 3.6 | SLIME | dec/opt |
| 17 | Dieta | 4.1 | PIXIE LP | opt |
| 18 | Minerales/carbón | 4.2 | PIXIE MIP | opt |
| 19 | Crudos no lineales | 4.3 | WMIBO/SLIME | dec/opt |
| 20 | Química/reactividad | 4.4 | SLIME/WMIBO | dec/opt |
| 21 | Nurse rostering | 5.1 | SLIME | dec |
| 22 | Deportes | 5.2 | SLIME/WMIBO | dec/opt |
| 23 | Covering arrays | 5.3 | BASILISK | count |
| 24 | Mapas procedurales | 5.4 | SLIME/BASILISK | dec/count |
| 25 | Música con reglas | 5.5 | SLIME/BASILISK | dec/count |
| 26 | Rompecabezas | 5.6 | SLIME/BASILISK | dec/count |
| 27 | Gaussianos | 5.7 | SLIME | dec |
| 28 | Cuántica inversa | 5.8 | SLIME | dec |
| 29 | Robot path | 5.9 | SLIME | dec |
| 30 | Menús | 5.10 | SLIME/WMIBO | dec/opt |
| 31 | CO₂ | 5.11 | WMIBO | opt |

---

## 7. Notas de rendimiento y buenas prácticas

- **Acotar siempre**: horizonte, granularidad de tiempo/rejilla y $B$ de
  presupuesto dominan el tamaño; empezar grueso y refinar (la solución
  gruesa guía la fina con suposiciones).
- **Simetrías**: muchos de estos problemas (muelles idénticos, máquinas
  idénticas, dígitos de Sudoku, vehículos homogéneos) tienen simetrías
  masivas; el pase de rompimiento de simetrías de la raíz (`use_symmetry`,
  activo por defecto) las elimina de forma sonora.
- **Conteo para dimensionar**: usar BASILISK (`count_projected`) para
  contar configuraciones admisibles (red de casilleros, mapas, diseños de
  experimento) en lugar de enumerar con un solver.
- **Blandas para negociar**: WMIBO permite declarar restricciones
  negociables con peso; el óptimo reporta qué se sacrificó y a qué costo.
- **Sesiones**: refinamientos interactivos (fijar una decisión, re-resolver
  sin reconstruir) mediante `satx::solver::session`; recordar que las
  sesiones no aplican rompimiento de simetrías (se fuerza `use_symmetry=0`).
- **Pruebas de UNSAT**: `--proof` genera DRAT para verificación independiente
  de infactibilidad (importante en diseño de circuitos y planificación
  legal).
