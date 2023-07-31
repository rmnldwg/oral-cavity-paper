# -*- coding: utf-8 -*-
"""
Plot histograms of dissected vs positive lymph nodes per LNL.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from lyscripts.plot.histograms import get_size
from matplotlib.colors import LinearSegmentedColormap
from shared import COLORS, DATAFILE, FIGURES_DIR, MPLSTYLE, load_and_prepare_data

OUTPUT_NAME = Path(__file__).with_suffix(".png").name
LNLS = ["Ib", "II", "III"]

positions = [0.0, 0.0001, 1.0]
colors = ["white", COLORS["green"], COLORS["red"]]
CMAP = LinearSegmentedColormap.from_list(
    "green_to_red",
    list(zip(positions, colors)),
    N=256,
)


def compute_min_max(LNLS, dataset):
    """Compute min and max values for x and y axes."""
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf

    for lnl in LNLS:
        total_dissected = dataset["total_dissected", "ipsi", lnl]
        positive_dissected = dataset["positive_dissected", "ipsi", lnl]

        xmin = np.minimum(xmin, total_dissected.min())
        xmax = np.maximum(xmax, total_dissected.max())
        xbins = np.arange(xmin, xmax + 1)

        ymin = np.minimum(ymin, positive_dissected.min())
        ymax = np.maximum(ymax, positive_dissected.max())
        ybins = np.arange(ymin, ymax + 1)

    return xmin, xmax, ymin, ymax, xbins, ybins


def add_text_to_hist2d(ax, xbins, ybins, hist2d, **kwargs):
    """Add text labels to 2D histogram."""
    text_kwargs = {
        "ha": "center",
        "va": "center",
        "color": "white",
        "fontsize": "x-small",
        "rotation": 90,
    }
    text_kwargs.update(kwargs)

    for k in range(hist2d.shape[0]):
        for j in range(hist2d.shape[1]):
            if hist2d[k, j] == 0:
                continue
            xpos, ypos = xbins[k] + 0.5, ybins[j] + 0.5
            label = f"{hist2d[k, j]:.0f}"
            ax.text(xpos, ypos, label, **text_kwargs)


if __name__ == "__main__":
    plt.style.use(MPLSTYLE)
    plt.rc(
        "axes",
        prop_cycle=cycler(color=[COLORS["red"], COLORS["orange"], COLORS["green"]]),
    )
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(
        nrows=6,
        ncols=2,
        sharex="col",
        sharey="row",
        figsize=get_size(width="full", ratio=1.0),
        constrained_layout=True,
        gridspec_kw={
            "width_ratios": [1.0, 0.16],
            "height_ratios": [1.0, 0.37, 1.0, 0.37, 1.0, 0.37],
            "hspace": 0.01,
            "wspace": 0.04,
        },
    )

    dataset, _ = load_and_prepare_data(filepath=DATAFILE, lnls=LNLS)
    xmin, xmax, ymin, ymax, xbins, ybins = compute_min_max(LNLS, dataset)

    for i, lnl in enumerate(LNLS):
        total_dissected = dataset["total_dissected", "ipsi", lnl]
        positive_dissected = dataset["positive_dissected", "ipsi", lnl]

        hist2d, *_ = ax[2 * i, 0].hist2d(
            total_dissected,
            positive_dissected,
            range=[(xmin - 0.5, xmax + 0.5), (ymin - 0.5, ymax + 0.5)],
            bins=(xbins, ybins),
            cmap=CMAP,
        )
        ax[2 * i, 0].set_title(f"LNL {lnl}")
        ax[2 * i, 0].set_ylabel("positive lymph nodes")
        ax[2 * i, 1].hist(
            positive_dissected,
            orientation="horizontal",
            range=(ymin - 0.5, ymax + 0.5),
            bins=ybins,
            color=COLORS["gray"],
        )
        ax[2 * i + 1, 0].hist(
            total_dissected,
            range=(xmin - 0.5, xmax + 0.5),
            bins=xbins,
            color=COLORS["gray"],
        )
        ax[2 * i + 1, 1].set_visible(False)
        ax[-1, 0].set_xlabel("dissected lymph nodes")

        add_text_to_hist2d(ax[2 * i, 0], xbins, ybins, hist2d)

    plt.savefig(FIGURES_DIR / OUTPUT_NAME)
