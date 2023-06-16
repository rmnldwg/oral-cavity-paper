# -*- coding: utf-8 -*-
"""
Create stacked bar plot of how often patients who presented with a certain combination
of involvements in the LNLs I, II, and III had extracapsular extension (ECE).
"""
# pylint: disable=import-error
# pylint: disable=singleton-comparison
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lyscripts.plot.histograms import get_size

from shared import DATAFILE, COLORS, load_and_prepare_data


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


def tf2str(tf: bool) -> str:
    """Transform `True` to `"pos"` and `False` to `"neg"`."""
    return "pos" if tf else "neg"


if __name__ == "__main__":
    dataset, max_llh_data = load_and_prepare_data(filepath=DATAFILE, lnls=LNLS)
    extracapsular = dataset["patient", "#", "extracapsular"]
    ipsi = max_llh_data["ipsi"]

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

    # Roman: I cannot reproduce these values. I get: 
    #   'ECE':      [10, 11, 2, 14, 10, 21, 5]
    #   'noECE':    [0 , 11, 2, 21, 5 , 30, 9]

    # ece = np.array([9, 11, 2, 14, 9, 21, 5])
    # noece = np.array([2, 11, 2, 22, 5, 31, 9])
    # ece_unk = np.array([0, 0, 0, 0, 0, 0, 0])

    # ece_ipsi = np.array([9, 10, 2, 12, 8, 15, 5])
    # noece_ipsi = np.array([2, 11, 2, 22, 5, 31, 9])
    # ece_ipsi_unk = np.array([0, 1, 0, 2, 1, 6, 0])

    # ece_contra = np.array([0, 0, 0, 0, 1, 0, 0])
    # noece_contra = np.array([11, 21, 4, 34, 12, 46, 14])
    # ece_contra_unk = np.array([0, 1, 0, 2, 1, 6, 0])

    plt.style.use(MPLSTYLE)
    fig, ax = plt.subplots(figsize=get_size())

    ece_bars = ax.bar(
        POSITIONS,
        counts["ECE"],
        label=f"ECE ({sum(counts['ECE'])})",
        color=COLORS["red"],
        width=WIDTHS,
    )
    noece_bars = ax.bar(
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

        ax.annotate(
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

        ax.annotate(
            text=f"{percent:.0f}%",
            xy=(x_loc, y_loc),
            ha="center",
            va="center",
            size=5,
            xytext=(0, 0),
            textcoords="offset points",
        )

    ax.set_xticks(POSITIONS)
    ax.set_xticklabels(labels, size=5)
    ax.set_xlabel("ipsilateral involvement")
    ax.set_ylabel("number of patients")
    ax.legend()
    ax.grid(axis="y")

    plt.savefig(OUTPUT_DIR / OUTPUT_NAME)
