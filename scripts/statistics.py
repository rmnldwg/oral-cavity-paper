"""
Plot panel with patient cohort statistics for the joined CLB & ISB dataset.
"""
# pylint: disable=singleton-comparison
from cProfile import label
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib import gridspec

from lyscripts.plot.histograms import get_size


DATAFILE = Path("./data/enhanced.csv")
OUTPUT_NAME = Path(__file__).with_suffix(".png").name
OUTPUT_DIR = Path("./figures")
MPLSTYLE = Path("./scripts/.mplstyle")

ORAL_CAVITY_ICD_CODES = {
    "tongue": ["C02", "C02.0", "C02.1", "C02.2", "C02.3", "C02.4", "C02.8", "C02.9",],
    "gums and cheeks": [
        "C03", "C03.0", "C03.1", "C03.9", "C06", "C06.0", "C06.1", "C06.2", "C06.8",
        "C06.9",
    ],
    "floor of mouth": ["C04", "C04.0", "C04.1", "C04.8", "C04.9",],
    "palate": ["C05", "C05.0", "C05.1", "C05.2", "C05.8", "C05.9",],
    "salivary glands": ["C08", "C08.0", "C08.1", "C08.9",],
}

# barplot settings
WIDTH, SPACE = 0.8, 0.6
LABELS  =          ["Ia"   , "Ib"   , "II" , "III", "IV" , "V"  ]
WIDTHS  = np.array([WIDTH/2, WIDTH/2, WIDTH, WIDTH, WIDTH, WIDTH])

# compute positions of bar centers based on WIDTHS and SPACE, such that the space
# between neighboring bars is SPACE. The first bar is centered at SPACE/2 + WIDTH/2.
POSITIONS = np.zeros_like(WIDTHS)
for i, width in enumerate(WIDTHS):
    spaces = (0.5 + i) * SPACE
    widths = sum(WIDTHS[:np.maximum(0,i)]) + width/2
    POSITIONS[i] = spaces + widths

COLORS = {
    "green": '#00afa5',
    "red": '#ae0060',
    "blue": '#005ea8',
    "orange": '#f17900',
    "gray": '#c5d5db',
}
COLOR_CYCLE = cycler(color=[COLORS["red"], COLORS["green"], COLORS["orange"]])


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
    plt.rc("axes", prop_cycle=COLOR_CYCLE)
    plt.rcParams['figure.constrained_layout.use'] = False

    dataset = pd.read_csv(DATAFILE, header=[0,1,2])
    is_oral_cavity = dataset["tumor", "1", "subsite"].isin(
        icd for icd_list in ORAL_CAVITY_ICD_CODES.values() for icd in icd_list
    )
    dataset = dataset.loc[is_oral_cavity]

    t_stages = dataset["tumor", "1", "t_stage"]
    subsites = dataset["tumor", "1", "subsite"]

    hpv_positive = dataset["patient", "#", "hpv_status"] == True
    hpv_negative = dataset["patient", "#", "hpv_status"] == False

    has_midext = dataset["tumor", "1", "extension"] == True
    has_not_midext = dataset["tumor", "1", "extension"] == False
    has_midext_unknown = dataset["tumor", "1", "extension"].isna()

    max_llh_data = dataset["max_llh"]

    num_total = len(max_llh_data)
    num_early = (t_stages <= 2).sum()
    num_late = (t_stages > 2).sum()

    # initialize figure
    fig = plt.figure(
        figsize=get_size(width="full", ratio=1./np.sqrt(2)),
    )

    # define grid layout
    gs = gridspec.GridSpec(ncols=2, nrows=3, figure=fig, wspace=0.075, hspace=0.15)
    ax = {}

    # define axes to plot on
    ax["prevalence contra"] = fig.add_subplot(gs[0,0])
    ax["prevalence ipsi"]   = fig.add_subplot(gs[0,1], sharey=ax["prevalence contra"])
    ax["row0"] = fig.add_subplot(gs[0,:], frame_on=False)

    ax["contra midext"]  = fig.add_subplot(gs[1,0])
    ax["contra ipsiI"] = fig.add_subplot(gs[1,1], sharey=ax["contra midext"])

    ax["subsites: tongue, gums, cheek"] = fig.add_subplot(gs[2,0])
    ax["subsites: floor of mouth, palate"] = fig.add_subplot(gs[2,1], sharey=ax["subsites: tongue, gums, cheek"])

    # first row, prevalence of involvement ipsi- & contralaterally
    ax["prevalence ipsi"].barh(
        POSITIONS,
        get_prevalence(max_llh_data["ipsi"], t_stages > 2, lnls=LABELS),
        label=f"T3 & T4 ({num_late})",
        height=WIDTHS
    )
    ax["prevalence ipsi"].barh(
        POSITIONS - SPACE/2.,
        get_prevalence(max_llh_data["ipsi"], t_stages <= 2, lnls=LABELS),
        label=f"T1 & T2 ({num_early})",
        height=WIDTHS
    )
    ax["prevalence ipsi"].scatter(
        get_prevalence(max_llh_data["ipsi"], lnls=LABELS),
        POSITIONS - SPACE/4.,
        s=300*WIDTHS,
        color="black",
        marker="|",
        label=f"total ({num_total})",
        zorder=5
    )
    y_lim = ax["prevalence ipsi"].get_ylim()
    ax["prevalence ipsi"].set_ylim(y_lim[::-1])
    ax["prevalence ipsi"].set_yticks(POSITIONS - SPACE/2.)
    ax["prevalence ipsi"].set_yticklabels(LABELS, ha="center", position=(-0.025,0))
    ax["prevalence ipsi"].grid(axis='y')
    ax["prevalence ipsi"].annotate(
        "ipsilateral", (0.5, 0.92),
        xycoords="axes fraction", horizontalalignment="center"
    )
    ax["prevalence ipsi"].legend(loc="lower right")
    x_lim = ax["prevalence ipsi"].get_xlim()

    ax["prevalence contra"].barh(
        POSITIONS,
        get_prevalence(max_llh_data["contra"], t_stages > 2, lnls=LABELS),
        label=f"T3 & T4 ({num_late})",
        height=WIDTHS
    )
    ax["prevalence contra"].barh(
        POSITIONS - SPACE/2.,
        get_prevalence(max_llh_data["contra"], t_stages <= 2, lnls=LABELS),
        label=f"T1 & T2 ({num_early})",
        height=WIDTHS
    )
    ax["prevalence contra"].scatter(
        get_prevalence(max_llh_data["contra"], lnls=LABELS),
        POSITIONS - SPACE/4,
        s=300*WIDTHS,
        color="black",
        marker="|",
        label=f"total ({num_total})",
        zorder=1.5,
    )
    ax["prevalence contra"].set_ylim(y_lim[::-1])
    ax["prevalence contra"].yaxis.tick_right()
    plt.setp(ax["prevalence contra"].get_yticklabels(), visible=False)
    ax["prevalence contra"].set_xlim(x_lim[::-1])
    ax["prevalence contra"].grid(axis='y')
    ax["prevalence contra"].annotate(
        "contralateral", (0.5, 0.92),
        xycoords="axes fraction", horizontalalignment="center"
    )
    ax["prevalence contra"].legend(loc="lower left")

    ax["row0"].set_xlabel("prevalence of involvement [%]", labelpad=9)
    ax["row0"].set_xticks([])
    ax["row0"].set_yticks([])

    # second row, contralateral involvement depending on midline extension and ipsilateral level III
    ax["contra midext"].bar(
        POSITIONS + SPACE/3.,
        get_prevalence(max_llh_data["contra"], has_midext, lnls=LABELS),
        label=f"midline extension ({sum(has_midext)})",
        width=WIDTHS,
    )
    ax["contra midext"].bar(
        POSITIONS - SPACE/3.,
        get_prevalence(max_llh_data["contra"], has_not_midext, lnls=LABELS),
        label=f"clearly lateralized ({sum(has_not_midext)})",
        width=WIDTHS,
        zorder=1.2,
    )
    ax["contra midext"].bar(
        POSITIONS,
        get_prevalence(max_llh_data["contra"], has_midext_unknown, lnls=LABELS),
        label=f"lateralization unknown ({sum(has_midext_unknown)})",
        width=WIDTHS,
    )
    ax["contra midext"].set_xticks(POSITIONS)
    ax["contra midext"].set_xticklabels(LABELS)
    ax["contra midext"].grid(axis='x')
    ax["contra midext"].set_ylabel("contralateral involvement [%]")
    ax["contra midext"].legend()

    has_ipsi_I = max_llh_data["ipsi", "I"] == True

    ax["contra ipsiI"].bar(
        POSITIONS,
        get_prevalence(max_llh_data["contra"], has_ipsi_I, lnls=LABELS),
        label=f"with involvement in LNL I ({sum(has_ipsi_I)})",
        width=WIDTHS,
    )
    ax["contra ipsiI"].bar(
        POSITIONS - SPACE/2.,
        get_prevalence(max_llh_data["contra"], ~has_ipsi_I, lnls=LABELS),
        label=f"without involvement in LNL I ({sum(~has_ipsi_I)})",
        width=WIDTHS,
    )
    ax["contra ipsiI"].set_xticks(POSITIONS - SPACE/2.)
    ax["contra ipsiI"].set_xticklabels(LABELS)
    ax["contra ipsiI"].grid(axis='x')
    ax["contra ipsiI"].legend()
    plt.setp(ax["contra ipsiI"].get_yticklabels(), visible=False)

    # third row, involvement by subsite
    ax["subsites: tongue, gums, cheek"].bar(
        POSITIONS,
        get_prevalence(
            max_llh_data["ipsi"],
            subsites.isin(ORAL_CAVITY_ICD_CODES["tongue"]),
            lnls=LABELS,
        ),
        label=f"tongue ({sum(subsites.isin(ORAL_CAVITY_ICD_CODES['tongue']))})",
        width=WIDTHS,
    )
    ax["subsites: tongue, gums, cheek"].bar(
        POSITIONS - SPACE/2.,
        get_prevalence(
            max_llh_data["ipsi"],
            subsites.isin(ORAL_CAVITY_ICD_CODES["gums and cheeks"]),
            lnls=LABELS,
        ),
        label=f"gums and cheek ({sum(subsites.isin(ORAL_CAVITY_ICD_CODES['gums and cheeks']))})",
        width=WIDTHS,
    )

    ax["subsites: floor of mouth, palate"].bar(
        POSITIONS,
        get_prevalence(
            max_llh_data["ipsi"],
            subsites.isin(ORAL_CAVITY_ICD_CODES["palate"]),
            lnls=LABELS,
        ),
        label=f"palate ({sum(subsites.isin(ORAL_CAVITY_ICD_CODES['palate']))})",
        width=WIDTHS,
    )
    ax["subsites: floor of mouth, palate"].bar(
        POSITIONS - SPACE/2.,
        get_prevalence(
            max_llh_data["ipsi"],
            subsites.isin(ORAL_CAVITY_ICD_CODES["floor of mouth"]),
            lnls=LABELS,
        ),
        label=f"floor of mouth ({sum(subsites.isin(ORAL_CAVITY_ICD_CODES['floor of mouth']))})",
        width=WIDTHS,
    )

    ax["subsites: tongue, gums, cheek"].set_xticks(POSITIONS - SPACE/2.)
    ax["subsites: tongue, gums, cheek"].set_xticklabels(LABELS)
    ax["subsites: tongue, gums, cheek"].grid(axis='x')
    ax["subsites: tongue, gums, cheek"].set_ylabel("subsite involvement [%]")
    ax["subsites: tongue, gums, cheek"].legend()

    ax["subsites: floor of mouth, palate"].set_xticks(POSITIONS - SPACE/2.)
    ax["subsites: floor of mouth, palate"].set_xticklabels(LABELS)
    ax["subsites: floor of mouth, palate"].grid(axis='x')
    ax["subsites: floor of mouth, palate"].legend()

    # labelling the six subplots
    ax["prevalence contra"].annotate("a)", (0.04, 0.92), xycoords="axes fraction")
    ax["prevalence ipsi"].annotate("b)", (0.96, 0.92), xycoords="axes fraction", horizontalalignment="right")
    ax["contra midext"].annotate("c)", (0.04, 0.92), xycoords="axes fraction")
    ax["contra ipsiI"].annotate("d)", (0.04, 0.92), xycoords="axes fraction")
    ax["subsites: tongue, gums, cheek"].annotate("e)", (0.04, 0.92), xycoords="axes fraction")
    ax["subsites: floor of mouth, palate"].annotate("f)", (0.04, 0.92), xycoords="axes fraction")

    plt.savefig(OUTPUT_DIR / OUTPUT_NAME)
