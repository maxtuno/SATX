cmake_minimum_required(VERSION 3.28)

project(satx VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SATX_BUILD_TESTS "Construir la suite de tests" ON)
option(SATX_BUILD_KERBEROS_CLI "Construir el despachador CLI de Kerberos" ON)

# ── satx: núcleo C++26 header-only ──
add_library(satx INTERFACE)
target_include_directories(satx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(satx INTERFACE cxx_std_26)

# ── Kerberos: kernel y backends C17, parte integral del producto satx ──
# Fuentes propias del repositorio en src/kerberos/ (sin third_party): SLIME (CDCL),
# BASILISK (#SAT), PIXIE, WMIBO, GRINDER y las capas de aceleración/paralelismo.
# Se compilan una sola vez en esta biblioteca y la comparten el puente C++
# (satx_solver) y el despachador CLI (kerberos).
set(SATX_KERBEROS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/kerberos")
if(NOT EXISTS "${SATX_KERBEROS_DIR}/slime.c")
  message(FATAL_ERROR "SATX_KERBEROS_DIR=${SATX_KERBEROS_DIR} no contiene slime.c")
endif()

add_library(satx_kerberos STATIC
  ${SATX_KERBEROS_DIR}/slime.c
  ${SATX_KERBEROS_DIR}/basilisk.c
  ${SATX_KERBEROS_DIR}/pixie.c
  ${SATX_KERBEROS_DIR}/wmibo.c
  ${SATX_KERBEROS_DIR}/grinder.c
  ${SATX_KERBEROS_DIR}/krb_accel.c
  ${SATX_KERBEROS_DIR}/krb_accel_cuda_stub.c
  ${SATX_KERBEROS_DIR}/krb_parallel_stub.c
)
target_include_directories(satx_kerberos PRIVATE ${SATX_KERBEROS_DIR})
target_compile_definitions(satx_kerberos PRIVATE
  SLIME_NO_MAIN BASILISK_NO_MAIN PIXIE_NO_MAIN WMIBO_NO_MAIN GRINDER_NO_MAIN)
set_target_properties(satx_kerberos PROPERTIES
  C_STANDARD 17
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)
# slime.c usa log2/sqrt/log/fmax: enlazar libm donde exista (Linux/MinGW).
if(NOT MSVC)
  target_link_libraries(satx_kerberos PUBLIC m)
endif()
if(WIN32)
  target_link_libraries(satx_kerberos PUBLIC advapi32)
endif()

# ── puente solver (C++26 → C API embebida de Kerberos) ──
add_library(satx_solver STATIC
  src/solver/kerberos.cpp
)
target_include_directories(satx_solver PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(satx_solver PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(satx_solver PUBLIC cxx_std_26)
target_link_libraries(satx_solver PUBLIC satx satx_kerberos)

# ── despachador CLI de Kerberos (SLIME/BASILISK/PIXIE/WMIBO + GRINDER) ──
if(SATX_BUILD_KERBEROS_CLI)
  add_executable(kerberos
    ${SATX_KERBEROS_DIR}/kerberos.c
  )
  target_include_directories(kerberos PRIVATE ${SATX_KERBEROS_DIR})
  target_link_libraries(kerberos PRIVATE satx_kerberos)
  set_target_properties(kerberos PROPERTIES
    C_STANDARD 17
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
  )
endif()

# ── tests (sin framework: asserts + bucles de propiedad) ──
# ADR-011: las aserciones se habilitan SIEMPRE (incluso en Release/NDEBUG);
# sin ellas la suite validaba nada.
enable_testing()

if(SATX_BUILD_TESTS)
  function(satx_add_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE satx satx_solver)
    target_compile_options(${name} PRIVATE
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
      $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()

  satx_add_test(test_negabinary)
  satx_add_test(test_complex_ops)
  satx_add_test(test_kerberos)
  satx_add_test(test_quantum_echoes)
  satx_add_test(test_solver_api)
endif()

# ── ejemplos ──
add_executable(template
  template.cpp
)
target_link_libraries(template PRIVATE satx satx_solver)

function(satx_add_example name)
  add_executable(${name} examples/${name}.cpp)
  target_link_libraries(${name} PRIVATE satx satx_solver)
endfunction()

satx_add_example(send_more_money)
satx_add_example(map_coloring)
satx_add_example(nqueens)
satx_add_example(rect_packing)
satx_add_example(optimize_sum)
satx_add_example(job_shop)
satx_add_example(dice_distribution)
satx_add_example(projectile)
satx_add_example(quantum_bell)
satx_add_example(quantum_learning)

# ── ejemplos avanzados (revisión ADR-011) ──
satx_add_example(gaussian_integer_factorization)
satx_add_example(mandelbrot_escape)
satx_add_example(quantum_teleportation)
satx_add_example(sudoku)
satx_add_example(model_counting)
satx_add_example(complex_polynomial_roots)

# ── instalación (distribución) ──
include(GNUInstallDirs)

install(TARGETS satx_kerberos satx_solver
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS template
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(SATX_BUILD_KERBEROS_CLI)
  install(TARGETS kerberos
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/satx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DOCDIR})
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
  install(FILES README.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
install(DIRECTORY docs/
  DESTINATION ${CMAKE_INSTALL_DOCDIR})
