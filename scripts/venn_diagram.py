"""
Create a Venn diagram of the involement in LNLs I, II, and III.
"""
# pylint: disable=singleton-comparison
from pathlib import Path
from itertools import product

import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

from lyscripts.plot.histograms import get_size

from shared import (
    DATAFILE,
    FIGURES_DIR,
    MPLSTYLE,
    COLORS,
    load_and_prepare_data,
)


OUTPUT_NAME = Path(__file__).with_suffix(".png").name


if __name__ == "__main__":
    plt.style.use(MPLSTYLE)
    plt.rc("axes", prop_cycle=cycler(color=[COLORS["red"], COLORS["orange"], COLORS["green"]]))
    plt.rcParams['figure.constrained_layout.use'] = True

    dataset, _ = load_and_prepare_data(filepath=DATAFILE, lnls=["I", "II", "III"])

    ipsi_data = dataset["max_llh", "ipsi"][["I", "II", "III"]]

    venn_data = {}
    for lnl_I, lnl_II, lnl_III in product([True, False], repeat=3):
        venn_data[(lnl_I, lnl_II, lnl_III)] = len(ipsi_data.loc[
            (ipsi_data["I"] == lnl_I) &
            (ipsi_data["II"] == lnl_II) &
            (ipsi_data["III"] == lnl_III)
        ])

    fig, ax = plt.subplots(figsize=get_size())

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
        subset_label_formatter=lambda x: f"{x}\n({x/len(ipsi_data):.0%})",
        ax=ax,
    )

    plt.savefig(FIGURES_DIR / OUTPUT_NAME)
