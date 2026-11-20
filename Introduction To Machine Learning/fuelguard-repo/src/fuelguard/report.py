"""Small helpers to export tables to LaTeX (no jinja2 dependency) and style figures."""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd

# Reference categorical palette (fixed order, validated for colour-vision deficiency)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLORS = {"Logistic (L2)": BLUE, "Random forest": ORANGE, "Gradient boosting": AQUA,
                "Logistic (implied vol only)": YELLOW}
FULL_MODELS = ["Logistic (L2)", "Random forest", "Gradient boosting"]


def style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 9.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
        "lines.linewidth": 1.6,
    })


def _fmt(v, fmt):
    if isinstance(v, str):
        return v.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", "{,}")
    return fmt.format(v)


def to_latex(df: pd.DataFrame, path, fmt: dict | None = None, default="{:.3f}",
             index_name: str = "", col_align: str | None = None, bold_rows=()):
    fmt = fmt or {}
    cols = list(df.columns)
    align = col_align or ("l" + "r" * len(cols))
    head = " & ".join([_fmt(index_name, "")] + [_fmt(str(c), "") for c in cols]) + r" \\"
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", head, r"\midrule"]
    for idx, row in df.iterrows():
        cells = [_fmt(str(idx), "")] + [_fmt(row[c], fmt.get(c, default)) for c in cols]
        if idx in bold_rows:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
