// sudoku — Sudoku 9×9 resuelto por SAT con la aritmética CBE.
//
// Cada celda es un complex<5,0> (parte imaginaria forzada a 0) con dominio
// 1..9; la restricción «todos distintos» de cada fila, columna y caja 3×3 se
// codifica con ¬eq_lit para cada par. Las pistas se imponen con eq_lit.
// El modelo SAT del kernel CDCL SLIME de Kerberos se decodifica y se
// verifica con una comprobación independiente en el host.

#include <satx/satx.hpp>

#include <array>
#include <cstdlib>
#include <iostream>

int main() {
  using C = satx::complex<5, 0>;  // caja NB [−10, 21] ⊇ {1..9}

  // Pista del Sudoku (0 = vacío).
  const std::array<std::array<int, 9>, 9> clues = {{
      {5, 3, 0, 0, 7, 0, 0, 0, 0},
      {6, 0, 0, 1, 9, 5, 0, 0, 0},
      {0, 9, 8, 0, 0, 0, 0, 6, 0},
      {8, 0, 0, 0, 6, 0, 0, 0, 3},
      {4, 0, 0, 8, 0, 3, 0, 0, 1},
      {7, 0, 0, 0, 2, 0, 0, 0, 6},
      {0, 6, 0, 0, 0, 0, 2, 8, 0},
      {0, 0, 0, 4, 1, 9, 0, 0, 5},
      {0, 0, 0, 0, 8, 0, 0, 7, 9},
  }};

  satx::engine e;
  std::array<std::array<C, 9>, 9> cell;
  for (int r = 0; r < 9; ++r) {
    for (int c = 0; c < 9; ++c) {
      cell[r][c] = C{e};
      // Parte imaginaria ≡ 0 (solo usamos el riel real).
      for (auto l : cell[r][c].im_pattern()) e.add_unit(-l);
      // Dominio 1..9.
      e.add_unit(satx::le_lit(e, C::one(e), cell[r][c]));
      e.add_unit(satx::le_lit(e, cell[r][c], C{9, 0}));
      // Pistas.
      if (clues[r][c] != 0) e.add_unit(satx::eq_lit(e, cell[r][c], C{clues[r][c], 0}));
    }
  }

  // Todos distintos en filas, columnas y cajas 3×3.
  auto pair_distinct = [&](const C& a, const C& b) {
    e.add_unit(-satx::eq_lit(e, a, b));
  };
  for (int r = 0; r < 9; ++r)
    for (int c1 = 0; c1 < 9; ++c1)
      for (int c2 = c1 + 1; c2 < 9; ++c2) pair_distinct(cell[r][c1], cell[r][c2]);
  for (int c = 0; c < 9; ++c)
    for (int r1 = 0; r1 < 9; ++r1)
      for (int r2 = r1 + 1; r2 < 9; ++r2) pair_distinct(cell[r1][c], cell[r2][c]);
  for (int br = 0; br < 3; ++br) {
    for (int bc = 0; bc < 3; ++bc) {
      for (int i = 0; i < 9; ++i) {
        for (int j = i + 1; j < 9; ++j) {
          pair_distinct(cell[br * 3 + i / 3][bc * 3 + i % 3],
                        cell[br * 3 + j / 3][bc * 3 + j % 3]);
        }
      }
    }
  }

  std::cout << "Resolviendo el Sudoku (variables: " << e.variable_count()
            << ", cláusulas: " << e.clause_count() << ")...\n";

  const auto m = satx::solver::solve(e);
  if (!m) {
    std::cerr << "UNSAT: el Sudoku no tiene solución.\n";
    return EXIT_FAILURE;
  }

  int grid[9][9]{};
  for (int r = 0; r < 9; ++r)
    for (int c = 0; c < 9; ++c) grid[r][c] = static_cast<int>(cell[r][c].value(*m).real());

  std::cout << "Solución:\n";
  for (int r = 0; r < 9; ++r) {
    for (int c = 0; c < 9; ++c) std::cout << grid[r][c] << ' ';
    std::cout << '\n';
  }

  // Verificación independiente en el host.
  for (int r = 0; r < 9; ++r) {
    std::array<bool, 10> seen{};
    for (int c = 0; c < 9; ++c) {
      const int v = grid[r][c];
      if (v < 1 || v > 9 || seen[v]) {
        std::cerr << "VERIFICACIÓN FALLIDA (fila " << r << ")\n";
        return EXIT_FAILURE;
      }
      seen[v] = true;
    }
  }
  for (int c = 0; c < 9; ++c) {
    std::array<bool, 10> seen{};
    for (int r = 0; r < 9; ++r) {
      const int v = grid[r][c];
      if (v < 1 || v > 9 || seen[v]) {
        std::cerr << "VERIFICACIÓN FALLIDA (columna " << c << ")\n";
        return EXIT_FAILURE;
      }
      seen[v] = true;
    }
  }
  for (int br = 0; br < 3; ++br) {
    for (int bc = 0; bc < 3; ++bc) {
      std::array<bool, 10> seen{};
      for (int i = 0; i < 9; ++i) {
        const int v = grid[br * 3 + i / 3][bc * 3 + i % 3];
        if (v < 1 || v > 9 || seen[v]) {
          std::cerr << "VERIFICACIÓN FALLIDA (caja)\n";
          return EXIT_FAILURE;
        }
        seen[v] = true;
      }
    }
  }
  for (int r = 0; r < 9; ++r)
    for (int c = 0; c < 9; ++c)
      if (clues[r][c] != 0 && grid[r][c] != clues[r][c]) {
        std::cerr << "VERIFICACIÓN FALLIDA (pista)\n";
        return EXIT_FAILURE;
      }
  std::cout << "Verificación del host: OK\n";
  return EXIT_SUCCESS;
}
