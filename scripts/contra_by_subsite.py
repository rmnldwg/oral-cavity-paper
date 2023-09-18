"""
Plot the contralateral involvement by subsite.
"""
from pathlib import Path
from cycler import cycler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lyscripts.plot.histograms import get_size

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
    lnls: list[str] | None = None,
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
    fig, ax = plt.subplots(figsize=get_size(ratio=1.0))

    dataset, max_llh_data = load_and_prepare_data(filepath=DATAFILE, lnls=LABELS)
    subsites = dataset["tumor", "1", "subsite"]

    ax.bar(
        POSITIONS + SPACE / 3.0,
        get_prevalence(
            max_llh_data["contra"],
            subsites.isin(ORAL_CAVITY_ICD_CODES["tongue"]),
            lnls=LABELS,
        ),
        label=f"tongue ({sum(subsites.isin(ORAL_CAVITY_ICD_CODES['tongue']))})",
        width=WIDTHS,
    )
    ax.bar(
        POSITIONS,
        get_prevalence(
            max_llh_data["contra"],
            subsites.isin(ORAL_CAVITY_ICD_CODES["floor of mouth"]),
            lnls=LABELS,
        ),
        label=f"floor of mouth ({sum(subsites.isin(ORAL_CAVITY_ICD_CODES['floor of mouth']))})",
        width=WIDTHS,
    )
    ax.bar(
        POSITIONS - SPACE / 3.0,
        get_prevalence(
            max_llh_data["contra"],
            subsites.isin(ORAL_CAVITY_ICD_CODES["gums & cheeks"]),
            lnls=LABELS,
        ),
        label=f"gums & cheeks ({sum(subsites.isin(ORAL_CAVITY_ICD_CODES['gums & cheeks']))})",
        width=WIDTHS,
    )

    ax.set_xticks(POSITIONS - SPACE / 3.0)
    ax.set_xticklabels(LABELS)
    ax.grid(axis="x")
    ax.set_ylabel("contralateral involvement [%]")
    ax.legend()

    plt.savefig(FIGURES_DIR / OUTPUT_NAME)
