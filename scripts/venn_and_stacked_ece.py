# -*- coding: utf-8 -*-
"""
Create a Venn diagram of co-involvements in LNLs I, II, and III in one panel, and a
stacked bar plot of how often patients who presented with a certain combination
of involvements in the LNLs I, II, and III had extracapsular extension (ECE) in a
second panel.
"""
# pylint: disable=import-error
# pylint: disable=singleton-comparison
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from cycler import cycler
import numpy as np
from lyscripts.plot.histograms import get_size
import pandas as pd

from shared import DATAFILE, COLORS, load_and_prepare_data, tf2str


OUTPUT_NAME = Path(__file__).with_suffix(".png").name
OUTPUT_DIR = Path("./figures")
MPLSTYLE = Path("./scripts/.mplstyle")


# barplot settings
WIDTH, SPACE = 0.8, 0.6
WIDTHS = np.array([WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH])

# compute positions of bar centers based on WIDTHS and SPACE, such that the space
# between neighboring bars is SPACE. The first bar is centered at SPACE/2 + WIDTH/2.
POSITIONS = np.zeros_like(WIDTHS)
for i, width in enumerate(WIDTHS):
    spaces = (0.5 + i) * SPACE
    widths = sum(WIDTHS[: np.maximum(0, i)]) + width / 2
    POSITIONS[i] = spaces + widths

LNLS = ["I", "II", "III"]


def prepare_venn_data(ipsi: pd.DataFrame):
    """Prepare data for Venn diagram."""
    venn_data = {}
    for lnl_I, lnl_II, lnl_III in product([True, False], repeat=3):
        venn_data[(lnl_I, lnl_II, lnl_III)] = len(
            ipsi.loc[
                (ipsi["I"] == lnl_I)
                & (ipsi["II"] == lnl_II)
                & (ipsi["III"] == lnl_III)
            ]
        )

    return venn_data


def plot_venn_diagram(ipsi, venn, venn_data):
    """Plot Venn diagram."""
    venn3(
        subsets=(
            venn_data[(True, False, False)],
            venn_data[(False, True, False)],
            venn_data[(True, True, False)],
            venn_data[(False, False, True)],
            venn_data[(True, False, True)],
            venn_data[(False, True, True)],
            venn_data[(True, True, True)],
        ),
        set_labels=("LNL I involved", "LNL II involved", "LNL III involved"),
        set_colors=(COLORS["orange"], COLORS["red"], COLORS["blue"]),
        alpha=0.6,
        subset_label_formatter=lambda x: f"{x}\n({x/len(ipsi):.0%})",
        ax=venn,
    )


def prepare_barplot_data(extracapsular, ipsi):
    """Prepare data for stacked ECE barplot."""
    labels = []
    counts = {"ECE": [], "noECE": [], "total": []}

    for inv_I, inv_II, inv_III in product([True, False], repeat=3):
        # skip N0 case
        if not inv_I and not inv_II and not inv_III:
            continue

        labels.append(
            f"I: {tf2str(inv_I)}\nII: {tf2str(inv_II)}\nIII: {tf2str(inv_III)}"
        )
        counts["ECE"].append(
            extracapsular[
                (ipsi["I"] == inv_I)
                & (ipsi["II"] == inv_II)
                & (ipsi["III"] == inv_III)
            ].sum()
        )
        counts["noECE"].append(
            (extracapsular[
                (ipsi["I"] == inv_I)
                & (ipsi["II"] == inv_II)
                & (ipsi["III"] == inv_III)
            ] != True).sum()
        )
        counts["total"].append(counts["ECE"][-1] + counts["noECE"][-1])

    return labels, counts


def plot_stacked_bars(WIDTHS, POSITIONS, labels, counts, bars):
    """Plot stacked ECE barplot."""
    ece_bars = bars.bar(
        POSITIONS,
        counts["ECE"],
        label=f"ECE ({sum(counts['ECE'])})",
        color=COLORS["red"],
        width=WIDTHS,
    )
    noece_bars = bars.bar(
        POSITIONS,
        counts["noECE"],
        bottom=counts["ECE"],
        label=f"no ECE ({sum(counts['noECE'])})",
        color=COLORS["green"],
        width=WIDTHS,
    )

    for patch, total in zip(ece_bars.patches, counts["total"]):
        x_loc = patch.get_x() + patch.get_width() / 2.0
        y_loc = patch.get_height() / 2.0 + patch.get_y() - 0.2
        percent = patch.get_height() / total * 100

        if percent == 0:
            continue

        bars.annotate(
            text=f"{percent:.0f}%",
            xy=(x_loc, y_loc),
            ha="center",
            va="center",
            size=5,
            xytext=(0, 0),
            textcoords="offset points",
        )

    for patch, total in zip(noece_bars.patches, counts["total"]):
        x_loc = patch.get_x() + patch.get_width() / 2.0
        y_loc = patch.get_height() / 2.0 + patch.get_y() - 0.2
        percent = patch.get_height() / total * 100

        if percent == 0:
            continue

        bars.annotate(
            text=f"{percent:.0f}%",
            xy=(x_loc, y_loc),
            ha="center",
            va="center",
            size=5,
            xytext=(0, 0),
            textcoords="offset points",
        )

    bars.set_xticks(POSITIONS)
    bars.set_xticklabels(labels, size=5)
    bars.set_xlabel("ipsilateral involvement")
    bars.set_ylabel("number of patients")
    bars.legend()
    bars.grid(axis="y")


if __name__ == "__main__":
    plt.style.use(MPLSTYLE)
    plt.rc(
        "axes",
        prop_cycle=cycler(color=[COLORS["red"], COLORS["orange"], COLORS["green"]]),
    )
    plt.rcParams["figure.constrained_layout.use"] = True

    dataset, max_llh_data = load_and_prepare_data(filepath=DATAFILE, lnls=LNLS)
    extracapsular = dataset["patient", "#", "extracapsular"]
    ipsi = max_llh_data["ipsi"]

    fig, (venn, bars) = plt.subplots(
        nrows=1, ncols=2,
        figsize=get_size(width="full", ratio=2.),
    )

    venn_data = prepare_venn_data(ipsi)
    plot_venn_diagram(ipsi, venn, venn_data)

    labels, counts = prepare_barplot_data(extracapsular, ipsi)
    plot_stacked_bars(WIDTHS, POSITIONS, labels, counts, bars)

    plt.savefig(OUTPUT_DIR / OUTPUT_NAME)
