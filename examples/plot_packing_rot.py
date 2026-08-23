#!/usr/bin/env python3
"""plot_packing_rot.py — grafica el problema y la solución del ejemplo
rect_packing_rot (empaquetado 2D CON rotación de 90°).

El ejemplo C++ examples/rect_packing_rot.cpp genera un JSON con los datos del
problema (contenedor y piezas con su columna inicial) y de la solución
(posiciones empaquetadas, orientación y qué piezas se colocaron, estilo
mochila). Este script dibuja ambos paneles con matplotlib; las piezas que
giraron se marcan con «↻».

Uso:
    python plot_packing_rot.py [archivo.json] [--save salida.png]
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Grafica el empaquetado 2D con rotación generado por rect_packing_rot.cpp"
    )
    ap.add_argument(
        "json",
        nargs="?",
        default="rect_packing_rot.json",
        help="archivo JSON generado por rect_packing_rot (default: rect_packing_rot.json)",
    )
    ap.add_argument("--save", metavar="PNG", default=None,
                    help="guardar la figura en un archivo PNG")
    args = ap.parse_args()

    try:
        with open(args.json, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        print(f"error: no se pudo leer '{args.json}': {e}", file=sys.stderr)
        return 1

    cont = data["contenedor"]
    W, H = cont["w"], cont["h"]
    rects = data["rectangulos"]
    sol = data["solucion"]
    colocados = {p["id"]: p for p in sol["colocados"]}

    colores = {r["id"]: plt.get_cmap("tab20")((r["id"] * 2 + 1) % 20)
               for r in rects}

    fig, (axp, axs) = plt.subplots(1, 2, figsize=(13, 6))

    # ── panel izquierdo: problema (contenedor + piezas sin colocar) ──
    # las piezas que al final NO se colocaron se muestran en gris; las que en
    # la solución quedaron rotadas llevan el marcador «↻»
    axp.add_patch(Rectangle((0, 0), W, H, fill=False, ec="black", lw=1.6))
    axp.text(W / 2, H / 2, f"{W}×{H}", ha="center", va="center",
             fontsize=16, color="0.55")
    for r in rects:
        p = colocados[r["id"]]
        colocado = p["colocado"]
        fc = colores[r["id"]] if colocado else "0.82"
        axp.add_patch(Rectangle((r["sx"], r["sy"]), r["w"], r["h"],
                                fc=fc, ec="black", alpha=0.85))
        etiqueta = f'{r["w"]}×{r["h"]}'
        if colocado and p["rotado"]:
            etiqueta += " ↻"
        axp.text(r["sx"] + r["w"] / 2, r["sy"] + r["h"] / 2,
                 etiqueta, ha="center", va="center", fontsize=8)
    axp.set_title(f'Problema — {len(rects)} piezas aleatorias\n'
                  f'contenedor {W}×{H} (semilla {data["semilla"]}, '
                  f'rotación 90° habilitada)')
    axp.set_xlabel("x")
    axp.set_ylabel("y")

    # ── panel derecho: solución (mayor cantidad que cabe, por SAT) ──
    # cada pieza se dibuja con sus dimensiones EFECTIVAS (girada o no);
    # «↻» marca las piezas que giraron en la solución
    axs.add_patch(Rectangle((0, 0), W, H, fill=False, ec="black", lw=1.6))
    for p in colocados.values():
        if not p["colocado"]:
            continue
        r = rects[p["id"]]
        axs.add_patch(Rectangle((p["x"], p["y"]), p["w_eff"], p["h_eff"],
                                fc=colores[r["id"]], ec="black", alpha=0.9))
        etiqueta = str(r["id"])
        if p["rotado"]:
            etiqueta += " ↻"
        axs.text(p["x"] + p["w_eff"] / 2, p["y"] + p["h_eff"] / 2,
                 etiqueta, ha="center", va="center", fontsize=9,
                 fontweight="bold")
    axs.text(W - 0.2, -0.7,
             f'{sol["objetivo"]} de {len(rects)} piezas   '
             f'área usada {sol["porcentaje_usado"]:.1f}%',
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    axs.set_title("Solución — mayor cantidad que cabe por SAT\n"
                  f'caja ocupada {sol["caja"]["w"]}×{sol["caja"]["h"]}, '
                  f'{sol["variables"]} vars, {sol["clausulas"]} cláusulas, '
                  f'{sol["resoluciones"]} resoluciones')
    axs.set_xlabel("x")
    axs.set_ylabel("y")

    # límites comunes: piezas del problema quedan encima del contenedor
    xmax = max(W, max(r["sx"] + r["w"] for r in rects))
    ymax = max(H, max(r["sy"] + r["h"] for r in rects))
    for ax in (axp, axs):
        ax.set_xlim(-0.8, xmax + 0.8)
        ax.set_ylim(-1.6, ymax + 0.8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"figura guardada en '{args.save}'")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
