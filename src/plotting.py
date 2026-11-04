"""Figure style shared by all notebooks (LaTeX text rendering, fixed colours)."""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")

# fixed categorical order; a method keeps its colour in every figure
COLORS = {
    "truth": "#1f1f1f",
    "C1-OLS": "#2a78d6",
    "C2-SINDy-FD": "#8fb8ea",
    "C2-SINDy-SG": "#eb6834",
    "C3-WSINDy": "#1baf7a",
    "N1-MLP": "#eda100",
    "N2-Lib": "#e87ba4",
    "N2-Lib-WS": "#6250d6",
}
LABELS = {
    "C1-OLS": r"C1 OLS (FD)",
    "C2-SINDy-FD": r"C2 SINDy (FD)",
    "C2-SINDy-SG": r"C2 SINDy (SG)",
    "C3-WSINDy": r"C3 weak SINDy",
    "N1-MLP": r"N1 black-box NODE",
    "N2-Lib": r"N2 sparse NODE",
    "N2-Lib-WS": r"N2-WS sparse NODE (warm)",
    "truth": r"ground truth",
}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9cc3ee", "#2a78d6", "#0d3a73"])


def setup(usetex=True):
    mpl.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath,amssymb,bm}",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
    })
    os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))


def clean_3d(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#dddddd")
        axis._axinfo["grid"]["color"] = "#eeeeee"
        axis._axinfo["grid"]["linewidth"] = 0.4
