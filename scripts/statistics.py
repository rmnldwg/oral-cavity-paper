"""
Plot panel with patient cohort statistics for the joined CLB & ISB dataset.
"""
from pathlib import Path
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

ORAL_CAVITY_ICD_CODES = [
    "C02", "C02.0", "C02.1", "C02.2", "C02.3", "C02.4", "C02.8", "C02.9",
    "C03", "C03.0", "C03.1", "C03.9",
    "C04", "C04.0", "C04.1", "C04.8", "C04.9",
    "C05", "C05.0", "C05.1", "C05.2", "C05.8", "C05.9",
    "C06", "C06.0", "C06.1", "C06.2", "C06.8", "C06.9",
    "C07",
    "C08", "C08.0", "C08.1", "C08.9",
]

# barplot settings
WIDTH, SPACE = 0.8, 0.4
LABELS  =          ["Ia"   , "Ib"   , "IIa"  , "IIb"  , "III", "IV" , "V"  ]
WIDTHS  = np.array([WIDTH/2, WIDTH/2, WIDTH/2, WIDTH/2, WIDTH, WIDTH, WIDTH])
POSITIONS = np.array([np.sum(WIDTHS[0:i])+i*SPACE for i in range(len(WIDTHS))])
POSITIONS[0] -= SPACE/2
POSITIONS[1] -= SPACE/2
POSITIONS[2] -= SPACE/2
POSITIONS[3] -= SPACE/2

COLORS = {
    "green": '#00afa5',
    "red": '#ae0060',
    "blue": '#005ea8',
    "orange": '#f17900',
    "gray": '#c5d5db',
}
COLOR_CYCLE = cycler(color=[COLORS["red"], COLORS["green"]])


if __name__ == "__main__":
    plt.style.use(MPLSTYLE)
    plt.rc("axes", prop_cycle=COLOR_CYCLE)
    plt.rcParams['figure.constrained_layout.use'] = False

    dataset = pd.read_csv(DATAFILE, header=[0,1,2])
    is_oral_cavity = dataset["tumor", "1", "subsite"].isin(ORAL_CAVITY_ICD_CODES)
    dataset = dataset.loc[is_oral_cavity].astype({
        ("tumor", "1", "central"): bool,
        ("tumor", "1", "extension"): bool,
    })


    t_stages = dataset["tumor", "1", "t_stage"]
    hpv_positive = dataset["patient", "#", "hpv_status"] == True
    hpv_negative = dataset["patient", "#", "hpv_status"] == False
    mid_ext = dataset["tumor", "1", "extension"]
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
    ax["contra ipsiIII"] = fig.add_subplot(gs[1,1], sharey=ax["contra midext"])

    ax["HPV early"] = fig.add_subplot(gs[2,0])
    ax["HPV late"]  = fig.add_subplot(gs[2,1], sharey=ax["HPV early"])

    # first row, prevalence of involvement ipsi- & contralaterally
    prev_ipsi = 100 * (max_llh_data["ipsi"] == True).sum() / num_total
    prev_ipsi_early = 100 * (max_llh_data["ipsi"] == True).loc[t_stages <= 2].sum() / num_early
    prev_ipsi_late = 100 * (max_llh_data["ipsi"] == True).loc[t_stages > 2].sum() / num_late
    ax["prevalence ipsi"].barh(
        POSITIONS,
        prev_ipsi_late[LABELS],
        label=f"T3 & T4 (ipsilateral, {num_late})",
        height=WIDTHS
    )
    ax["prevalence ipsi"].barh(
        POSITIONS - SPACE/2.,
        prev_ipsi_early[LABELS],
        label=f"T1 & T2 (ipsilateral, {num_early})",
        height=WIDTHS
    )
    ax["prevalence ipsi"].scatter(
        prev_ipsi[LABELS],
        POSITIONS - SPACE/4,
        s=300*WIDTHS,
        color="black",
        marker="|",
        label=f"total (ipsilateral, {num_total})",
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

    prev_contra = (100 / num_total) * (
        max_llh_data["contra"] == True
    ).sum()
    prev_contra_early = (100 / num_early) * (
        max_llh_data["contra"] == True
    ).loc[t_stages <= 2].sum()
    prev_contra_late = (100 / num_late) * (
        max_llh_data["contra"] == True
    ).loc[t_stages > 2].sum()
    ax["prevalence contra"].barh(
        POSITIONS,
        prev_contra_late[LABELS],
        label=f"T3 & T4 (contralateral, {num_late})",
        height=WIDTHS
    )
    ax["prevalence contra"].barh(
        POSITIONS - SPACE/2.,
        prev_contra_early[LABELS],
        label=f"T1 & T2 (contralateral, {num_early})",
        height=WIDTHS
    )
    ax["prevalence contra"].scatter(
        prev_contra[LABELS],
        POSITIONS - SPACE/4,
        s=300*WIDTHS,
        color="black",
        marker="|",
        label=f"total (contralateral, {num_total})",
        zorder=5
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
    num_midext = len(max_llh_data[mid_ext])
    num_nomidext = len(max_llh_data[~mid_ext])
    contra_midext = (100 / num_midext) * (
        max_llh_data["contra"] == True
    ).loc[mid_ext].sum()
    contra_nomidext = (100 / num_nomidext) * (
        max_llh_data["contra"] == True
    ).loc[~mid_ext].sum()

    ax["contra midext"].bar(
        POSITIONS,
        contra_midext[LABELS],
        label=f"with midline extension ({num_midext})",
        width=WIDTHS
    )
    ax["contra midext"].bar(
        POSITIONS - SPACE/2.,
        contra_nomidext[LABELS],
        label=f"without midline extension ({num_nomidext})",
        width=WIDTHS
    )
    ax["contra midext"].set_xticks(POSITIONS - SPACE/2.)
    ax["contra midext"].set_xticklabels(LABELS)
    ax["contra midext"].grid(axis='x')
    ax["contra midext"].set_ylabel("contralateral involvement [%]")
    ax["contra midext"].legend()

    num_ipsiIII = (max_llh_data["ipsi", "III"] == True).sum()
    num_noipsiIII = (max_llh_data["ipsi", "III"] != True).sum()

    contra_ipsiIII = (100 / num_ipsiIII) * (
        max_llh_data["contra"] == True
    ).loc[
        max_llh_data["ipsi", "III"] == True
    ].sum()

    contra_noipsiIII = (100 / num_noipsiIII) * (
        max_llh_data["contra"] == True
    ).loc[
        max_llh_data["ipsi", "III"] != True
    ].sum()

    ax["contra ipsiIII"].bar(
        POSITIONS,
        contra_ipsiIII[LABELS],
        label=f"with involvement in LNL III ({num_ipsiIII})",
        width=WIDTHS
    )
    ax["contra ipsiIII"].bar(
        POSITIONS - SPACE/2.,
        contra_noipsiIII[LABELS],
        label=f"without involvement in LNL III ({num_noipsiIII})",
        width=WIDTHS
    )
    ax["contra ipsiIII"].set_xticks(POSITIONS - SPACE/2.)
    ax["contra ipsiIII"].set_xticklabels(LABELS)
    ax["contra ipsiIII"].grid(axis='x')
    ax["contra ipsiIII"].legend()
    plt.setp(ax["contra ipsiIII"].get_yticklabels(), visible=False)

    # third row, HPV positive vs negative
    num_HPVpos_early = len(max_llh_data.loc[hpv_positive & (t_stages <= 2)])
    num_HPVneg_early = len(max_llh_data.loc[hpv_negative & (t_stages <= 2)])
    ipsi_HPVpos_early = (100 / num_HPVpos_early) * (
        max_llh_data["ipsi"] == True
    ).loc[hpv_positive & (t_stages <= 2)].sum()
    ipsi_HPVneg_early = (100 / num_HPVneg_early) * (
        max_llh_data["ipsi"] == True
    ).loc[hpv_negative & (t_stages <= 2)].sum()

    ax["HPV early"].bar(
        POSITIONS,
        ipsi_HPVpos_early[LABELS],
        label=f"HPV+ ({num_HPVpos_early})",
        width=WIDTHS
    )
    ax["HPV early"].bar(
        POSITIONS - SPACE/2.,
        ipsi_HPVneg_early[LABELS],
        label=f"HPV- ({num_HPVneg_early})",
        width=WIDTHS
    )
    ax["HPV early"].set_xticks(POSITIONS - SPACE/2.)
    ax["HPV early"].set_xticklabels(LABELS)
    ax["HPV early"].set_ylabel("ipsilateral involvement [%]")
    ax["HPV early"].grid(axis='x')
    ax["HPV early"].annotate(
        "T1 & T2", (0.5, 0.92),
        xycoords="axes fraction", horizontalalignment="center"
    )
    ax["HPV early"].legend()

    num_HPVpos_late = len(max_llh_data.loc[hpv_positive & (t_stages > 2)])
    num_HPVneg_late = len(max_llh_data.loc[hpv_negative & (t_stages > 2)])
    ipsi_HPVpos_late = (100 / num_HPVpos_late) * (
        max_llh_data["ipsi"] == True
    ).loc[hpv_positive & (t_stages > 2)].sum()
    ipsi_HPVneg_late = (100 / num_HPVneg_late) * (
        max_llh_data["ipsi"] == True
    ).loc[hpv_negative & (t_stages > 2)].sum()

    ax["HPV late"].bar(
        POSITIONS,
        ipsi_HPVpos_late[LABELS],
        label=f"HPV+ ({num_HPVpos_late})",
        width=WIDTHS
    )
    ax["HPV late"].bar(
        POSITIONS - SPACE/2.,
        ipsi_HPVneg_late[LABELS],
        label=f"HPV- ({num_HPVneg_late})",
        width=WIDTHS
    )
    ax["HPV late"].set_xticks(POSITIONS - SPACE/2.)
    ax["HPV late"].set_xticklabels(LABELS)
    ax["HPV late"].grid(axis='x')
    plt.setp(ax["HPV late"].get_yticklabels(), visible=False)
    ax["HPV late"].annotate(
        "T3 & T4", (0.5, 0.92),
        xycoords="axes fraction", horizontalalignment="center"
    )
    ax["HPV late"].legend()

    # labelling the six subplots
    ax["prevalence contra"].annotate("a)", (0.04, 0.92), xycoords="axes fraction")
    ax["prevalence ipsi"].annotate("b)", (0.96, 0.92), xycoords="axes fraction", horizontalalignment="right")
    ax["contra midext"].annotate("c)", (0.04, 0.92), xycoords="axes fraction")
    ax["contra ipsiIII"].annotate("d)", (0.04, 0.92), xycoords="axes fraction")
    ax["HPV early"].annotate("e)", (0.04, 0.92), xycoords="axes fraction")
    ax["HPV late"].annotate("f)", (0.04, 0.92), xycoords="axes fraction")

    plt.savefig(OUTPUT_DIR / OUTPUT_NAME)
