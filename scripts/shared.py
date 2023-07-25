# -*- coding: utf-8 -*-
"""
Some shared params, functions, and classes.
"""
from pathlib import Path
from typing import List

import pandas as pd

DATA_DIR = Path("./data")
DATAFILE = DATA_DIR / "enhanced.csv"
FIGURES_DIR = Path("./figures")
TABLES_DIR = Path("./tables")
SCRIPTS_DIR = Path("./scripts")

MPLSTYLE = SCRIPTS_DIR / ".mplstyle"
COLORS = {
    "green": "#00afa5",
    "red": "#ae0060",
    "blue": "#005ea8",
    "orange": "#f17900",
    "gray": "#c5d5db",
}

ORAL_CAVITY_ICD_CODES = {
    "tongue": [
        "C02",
        "C02.0",
        "C02.1",
        "C02.2",
        "C02.3",
        "C02.4",
        "C02.8",
        "C02.9",
    ],
    "gums & cheeks": [
        "C03",
        "C03.0",
        "C03.1",
        "C03.9",
        "C06",
        "C06.0",
        "C06.1",
        "C06.2",
        "C06.8",
        "C06.9",
    ],
    "floor of mouth": [
        "C04",
        "C04.0",
        "C04.1",
        "C04.8",
        "C04.9",
    ],
    # "palate": ["C05", "C05.0", "C05.1", "C05.2", "C05.8", "C05.9",],
    # "salivary glands": ["C08", "C08.0", "C08.1", "C08.9",],
}


def add_percent(columns: List[str]) -> List[str]:
    """Add a column with percentages to the given columns."""
    result_columns = []
    for col in columns:
        result_columns.append(col)
        if isinstance(col, tuple):
            *tmp, last = col
            result_columns.append((*tmp, last + "%"))
        else:
            result_columns.append(col + "%")
    return result_columns


def tf2str(tf: bool) -> str:
    """Transform `True` to `"pos"` and `False` to `"neg"`."""
    return "pos" if tf else "neg"


def load_and_prepare_data(filepath: Path | str, lnls: list[str]):
    """Load data from `filepath` and prepare it for further analysis."""
    dataset = pd.read_csv(Path(filepath), header=[0, 1, 2])
    is_oral_cavity = dataset["tumor", "1", "subsite"].isin(
        icd for icd_list in ORAL_CAVITY_ICD_CODES.values() for icd in icd_list
    )
    dataset = dataset.loc[is_oral_cavity]
    max_llh_data = dataset["max_llh"]

    cols_to_drop = []
    for lnl in max_llh_data.columns.get_level_values(1):
        if lnl not in lnls:
            cols_to_drop.append(("ipsi", lnl))
            cols_to_drop.append(("contra", lnl))

    max_llh_data = max_llh_data.drop(columns=cols_to_drop)
    return dataset, max_llh_data
