# -*- coding: utf-8 -*-
"""
Plot panel with patient cohort statistics for the joined CLB & ISB dataset.
"""
# pylint: disable=singleton-comparison
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from lyscripts.plot.histograms import get_size
from matplotlib import gridspec
from shared import (
    COLORS,
    DATAFILE,
    FIGURES_DIR,
    MPLSTYLE,
    ORAL_CAVITY_ICD_CODES,
    load_and_prepare_data,
)

OUTPUT_NAME = Path(__file__).with_suffix(".png").name

# barplot settings
WIDTH, SPACE = 0.8, 0.6
LABELS = ["Ia", "Ib", "II", "III", "IV", "V"]
WIDTHS = np.array([WIDTH / 2, WIDTH / 2, WIDTH, WIDTH, WIDTH, WIDTH])

# compute positions of bar centers based on WIDTHS and SPACE, such that the space
# between neighboring bars is SPACE. The first bar is centered at SPACE/2 + WIDTH/2.
POSITIONS = np.zeros_like(WIDTHS)
for i, width in enumerate(WIDTHS):
    spaces = (0.5 + i) * SPACE
    widths = sum(WIDTHS[: np.maximum(0, i)]) + width / 2
    POSITIONS[i] = spaces + widths


def get_prevalence(
    data: pd.DataFrame,
    locs: pd.Series | None = None,
    lnls: List[str] | None = None,
) -> pd.Series:
    """Compute the prevalence of lymphatic involvement in any given LNL in percent.

    Args:
        data: Table with rows of patients and columns of LNLs.
        locs: Indices of patients to consider.
        lnls: LNLs to consider.
    """
    num = len(data)
    if locs is not None:
        data = data.loc[locs]
        num = sum(locs)

    prevalences = 100 * (data == True).sum() / num
    return prevalences[lnls] if lnls else prevalences


if __name__ == "__main__":
    plt.style.use(MPLSTYLE)
    plt.rc(
        "axes",
        prop_cycle=cycler(color=[COLORS["red"], COLORS["orange"], COLORS["green"]]),
    )
    plt.rcParams["figure.constrained_layout.use"] = False

    dataset, max_llh_data = load_and_prepare_data(filepath=DATAFILE, lnls=LABELS)

    t_stages = dataset["tumor", "1", "t_stage"]
    subsites = dataset["tumor", "1", "subsite"]

    hpv_positive = dataset["patient", "#", "hpv_status"] == True
    hpv_negative = dataset["patient", "#", "hpv_status"] == False

    extracapsular = dataset["patient", "#", "extracapsular"]

    has_midext = dataset["tumor", "1", "extension"] == True
    has_not_midext = dataset["tumor", "1", "extension"] == False
    has_midext_unknown = dataset["tumor", "1", "extension"].isna()

    is_n0 = max_llh_data.sum(axis=1) == 0

    num_total = len(max_llh_data)
    num_early = (t_stages <= 2).sum()
    num_late = (t_stages > 2).sum()

    # initialize figure
    fig = plt.figure(
        figsize=get_size(width="full", ratio=1.0 / np.sqrt(2)),
    )

    # define grid layout
    gs = gridspec.GridSpec(ncols=2, nrows=3, figure=fig, wspace=0.075, hspace=0.15)
    ax = {}

    # define axes to plot on
    ax["prevalence contra"] = fig.add_subplot(gs[0, 0])
    ax["prevalence ipsi"] = fig.add_subplot(gs[0, 1], sharey=ax["prevalence contra"])
    ax["top row"] = fig.add_subplot(gs[0, :], frame_on=False)

    ax["contra midext"] = fig.add_subplot(gs[1, 0])
    ax["contra ipsi"] = fig.add_subplot(gs[1, 1], sharey=ax["contra midext"])

    ax["subsites"] = fig.add_subplot(gs[2, 0])
    ax["extracapsular"] = fig.add_subplot(gs[2, 1])

    # first row, prevalence of involvement ipsi- & contralaterally
    ax["prevalence ipsi"].barh(
        POSITIONS,
        get_prevalence(max_llh_data["ipsi"], t_stages > 2, lnls=LABELS),
        label=f"T3 & T4 ({num_late})",
        height=WIDTHS,
        color=COLORS["red"],
    )
    ax["prevalence ipsi"].barh(
        POSITIONS - SPACE / 2.0,
        get_prevalence(max_llh_data["ipsi"], t_stages <= 2, lnls=LABELS),
        label=f"T1 & T2 ({num_early})",
        height=WIDTHS,
        color=COLORS["green"],
    )
    ax["prevalence ipsi"].scatter(
        get_prevalence(max_llh_data["ipsi"], lnls=LABELS),
        POSITIONS - SPACE / 4.0,
        s=300 * WIDTHS,
        color="black",
        marker="|",
        label=f"total ({num_total})",
        zorder=5,
    )
    y_lim = ax["prevalence ipsi"].get_ylim()
    ax["prevalence ipsi"].set_ylim(y_lim[::-1])
    ax["prevalence ipsi"].set_yticks(POSITIONS - SPACE / 2.0)
    ax["prevalence ipsi"].set_yticklabels(LABELS, ha="center", position=(-0.025, 0))
    ax["prevalence ipsi"].grid(axis="y")
    ax["prevalence ipsi"].annotate(
        "ipsilateral",
        (0.5, 0.92),
        xycoords="axes fraction",
        horizontalalignment="center",
    )
    ax["prevalence ipsi"].legend(loc="lower right")
    x_lim = ax["prevalence ipsi"].get_xlim()

    ax["prevalence contra"].barh(
        POSITIONS,
        get_prevalence(max_llh_data["contra"], t_stages > 2, lnls=LABELS),
        label=f"T3 & T4 ({num_late})",
        height=WIDTHS,
        color=COLORS["red"],
    )
    ax["prevalence contra"].barh(
        POSITIONS - SPACE / 2.0,
        get_prevalence(max_llh_data["contra"], t_stages <= 2, lnls=LABELS),
        label=f"T1 & T2 ({num_early})",
        height=WIDTHS,
        color=COLORS["green"],
    )
    ax["prevalence contra"].scatter(
        get_prevalence(max_llh_data["contra"], lnls=LABELS),
        POSITIONS - SPACE / 4,
        s=300 * WIDTHS,
        color="black",
        marker="|",
        label=f"total ({num_total})",
        zorder=1.5,
    )
    ax["prevalence contra"].set_ylim(y_lim[::-1])
    ax["prevalence contra"].yaxis.tick_right()
    plt.setp(ax["prevalence contra"].get_yticklabels(), visible=False)
    ax["prevalence contra"].set_xlim(x_lim[::-1])
    ax["prevalence contra"].grid(axis="y")
    ax["prevalence contra"].annotate(
        "contralateral",
        (0.5, 0.92),
        xycoords="axes fraction",
        horizontalalignment="center",
    )
    ax["prevalence contra"].legend(loc="lower left")

    ax["top row"].set_xlabel("prevalence of involvement [%]", labelpad=9)
    ax["top row"].set_xticks([])
    ax["top row"].set_yticks([])

    # second row, contralateral involvement depending on midline extension and ipsilateral level III
    ax["contra midext"].bar(
        POSITIONS + SPACE / 3.0,
        get_prevalence(max_llh_data["contra"], has_midext, lnls=LABELS),
        label=f"midline extension ({sum(has_midext)})",
        width=WIDTHS,
    )
    ax["contra midext"].bar(
        POSITIONS,
        get_prevalence(max_llh_data["contra"], has_midext_unknown, lnls=LABELS),
        label=f"lateralization unknown ({sum(has_midext_unknown)})",
        width=WIDTHS,
    )
    ax["contra midext"].bar(
        POSITIONS - SPACE / 3.0,
        get_prevalence(max_llh_data["contra"], has_not_midext, lnls=LABELS),
        label=f"clearly lateralized ({sum(has_not_midext)})",
        width=WIDTHS,
        zorder=1.2,
    )
    ax["contra midext"].set_xticks(POSITIONS - SPACE / 3.0)
    ax["contra midext"].set_xticklabels(LABELS)
    ax["contra midext"].grid(axis="x")
    ax["contra midext"].set_ylabel("contralateral involvement [%]")
    ax["contra midext"].legend()

    is_ipsi_n0 = max_llh_data["ipsi"].sum(axis=1) == 0
    has_ipsi_one = max_llh_data["ipsi"].sum(axis=1) == 1
    has_ipsi_more = max_llh_data["ipsi"].sum(axis=1) > 1

    ax["contra ipsi"].bar(
        POSITIONS + SPACE / 3.0,
        get_prevalence(max_llh_data["contra"], has_ipsi_more, lnls=LABELS),
        label=f"2 or more ipsi LNLs ({sum(has_ipsi_more)})",
        width=WIDTHS,
    )
    ax["contra ipsi"].bar(
        POSITIONS,
        get_prevalence(max_llh_data["contra"], has_ipsi_one, lnls=LABELS),
        label=f"one ipsi LNL ({sum(has_ipsi_one)})",
        width=WIDTHS,
    )
    ax["contra ipsi"].bar(
        POSITIONS - SPACE / 3.0,
        get_prevalence(max_llh_data["contra"], is_ipsi_n0, lnls=LABELS),
        label=f"ipsi N0 ({sum(is_ipsi_n0)})",
        width=WIDTHS,
    )

    ax["contra ipsi"].set_xticks(POSITIONS - SPACE / 3.0)
    ax["contra ipsi"].set_xticklabels(LABELS)
    ax["contra ipsi"].grid(axis="x")
    ax["contra ipsi"].legend()
    plt.setp(ax["contra ipsi"].get_yticklabels(), visible=False)

    # third row, involvement by subsite
    ax["subsites"].bar(
        POSITIONS + SPACE / 3.0,
        get_prevalence(
            max_llh_data["ipsi"],
            subsites.isin(ORAL_CAVITY_ICD_CODES["tongue"]),
            lnls=LABELS,
        ),
        label=f"tongue ({sum(subsites.isin(ORAL_CAVITY_ICD_CODES['tongue']))})",
        width=WIDTHS,
    )
    ax["subsites"].bar(
        POSITIONS,
        get_prevalence(
            max_llh_data["ipsi"],
            subsites.isin(ORAL_CAVITY_ICD_CODES["floor of mouth"]),
            lnls=LABELS,
        ),
        label=f"floor of mouth ({sum(subsites.isin(ORAL_CAVITY_ICD_CODES['floor of mouth']))})",
        width=WIDTHS,
    )
    ax["subsites"].bar(
        POSITIONS - SPACE / 3.0,
        get_prevalence(
            max_llh_data["ipsi"],
            subsites.isin(ORAL_CAVITY_ICD_CODES["gums and cheeks"]),
            lnls=LABELS,
        ),
        label=f"gums and cheek ({sum(subsites.isin(ORAL_CAVITY_ICD_CODES['gums and cheeks']))})",
        width=WIDTHS,
    )

    ax["subsites"].set_xticks(POSITIONS - SPACE / 3.0)
    ax["subsites"].set_xticklabels(LABELS)
    ax["subsites"].grid(axis="x")
    ax["subsites"].set_ylabel("subsite involvement [%]")
    ax["subsites"].legend()

    # fourth row, extracapsular involvement
    is_not_n0_and_has_not_ece = (extracapsular == False) & ~is_n0
    has_ece = extracapsular == True

    ax["extracapsular"].bar(
        POSITIONS + SPACE / 2.0,
        get_prevalence(max_llh_data["ipsi"], has_ece, lnls=LABELS),
        label=f"extracapsular extension ({sum(has_ece)})",
        width=WIDTHS,
    )
    ax["extracapsular"].bar(
        POSITIONS,
        get_prevalence(max_llh_data["ipsi"], is_not_n0_and_has_not_ece, lnls=LABELS),
        label=f"no extracapsular extension ({sum(is_not_n0_and_has_not_ece)})",
        width=WIDTHS,
        color=COLORS["green"],
    )

    ax["extracapsular"].set_xticks(POSITIONS)
    ax["extracapsular"].set_xticklabels(LABELS)
    ax["extracapsular"].grid(axis="x")
    ax["extracapsular"].yaxis.set_label_position("right")
    ax["extracapsular"].yaxis.tick_right()
    ax["extracapsular"].set_ylabel("involvement [%]")
    ax["extracapsular"].legend()

    # labelling the six subplots
    ax["prevalence contra"].annotate("a)", (0.04, 0.92), xycoords="axes fraction")
    ax["prevalence ipsi"].annotate(
        "b)", (0.96, 0.92), xycoords="axes fraction", horizontalalignment="right"
    )
    ax["contra midext"].annotate("c)", (0.04, 0.92), xycoords="axes fraction")
    ax["contra ipsi"].annotate("d)", (0.04, 0.92), xycoords="axes fraction")
    ax["subsites"].annotate("e)", (0.04, 0.92), xycoords="axes fraction")
    ax["extracapsular"].annotate("f)", (0.04, 0.92), xycoords="axes fraction")

    plt.savefig(FIGURES_DIR / OUTPUT_NAME)
