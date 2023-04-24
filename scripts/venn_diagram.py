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
    # "palate": ["C05", "C05.0", "C05.1", "C05.2", "C05.8", "C05.9",],
    # "salivary glands": ["C08", "C08.0", "C08.1", "C08.9",],
}

COLORS = {
    "green": '#00afa5',
    "red": '#ae0060',
    "blue": '#005ea8',
    "orange": '#f17900',
    "gray": '#c5d5db',
}


if __name__ == "__main__":
    plt.style.use(MPLSTYLE)
    plt.rc("axes", prop_cycle=cycler(color=[COLORS["red"], COLORS["orange"], COLORS["green"]]))
    plt.rcParams['figure.constrained_layout.use'] = True

    dataset = pd.read_csv(DATAFILE, header=[0,1,2])
    is_oral_cavity = dataset["tumor", "1", "subsite"].isin(
        icd for icd_list in ORAL_CAVITY_ICD_CODES.values() for icd in icd_list
    )
    dataset = dataset.loc[is_oral_cavity]

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

    plt.savefig(OUTPUT_DIR / OUTPUT_NAME)
