"""Identidade visual dos gráficos do curso.

Um único ponto de configuração. Os notebooks chamam :func:`estilo` na primeira
célula e depois usam matplotlib puro — sem escolher cor, fonte ou grade caso a
caso.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Paleta do curso, em ordem de uso. Azul-petróleo para o caso base, terracota
#: para o contraste, cinza para referências e anotações.
PALETA: list[str] = [
    "#1f4e5f",
    "#c1553b",
    "#7a8b8b",
    "#d9a441",
    "#4a6fa5",
    "#8c6d8f",
]

TINTA = "#2b2b2b"
GRADE = "#d8d5cf"


def estilo() -> None:
    """Aplica o estilo do curso ao matplotlib global."""
    mpl.rcParams.update(
        {
            "figure.figsize": (7.2, 4.2),
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.prop_cycle": mpl.cycler(color=PALETA),
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9.5,
            "axes.labelcolor": TINTA,
            "axes.edgecolor": GRADE,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRADE,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "xtick.color": TINTA,
            "ytick.color": TINTA,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "text.color": TINTA,
        }
    )


def fonte(ax: plt.Axes, texto: str) -> None:
    """Escreve uma nota de rodapé discreta abaixo do eixo."""
    ax.annotate(
        texto,
        xy=(0, -0.18),
        xycoords="axes fraction",
        fontsize=8,
        color="#6f6a63",
        annotation_clip=False,
    )
