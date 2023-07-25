# -*- coding: utf-8 -*-
"""
Simultaneous involvement in levels I, II, and III for the whole patient cohort (all),
stratified according to early (T1/T2) versus advanced (T3/T4) T-category, and
stratified according to ECE. Columns 1-3 define the 8 possible combinations of
involvement; subsequent columns report the number of patients with the respective
combination of co-involved levels.
"""
# pylint: disable=import-error
# pylint: disable=singleton-comparison
from itertools import product
from pathlib import Path
import warnings

import pandas as pd
from shared import DATAFILE, TABLES_DIR, load_and_prepare_data, tf2str

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

OUTPUT_NAME = Path(__file__).with_suffix(".csv").name
LNLS = ["I", "II", "III", "IV", "V"]


if __name__ == "__main__":
    dataset, max_llh_data = load_and_prepare_data(filepath=DATAFILE, lnls=LNLS)

    is_nplus = max_llh_data.sum(axis=1) != 0
    has_extracapsular = dataset["patient", "#", "extracapsular"] == True
    has_no_extracapsular = dataset["patient", "#", "extracapsular"] == False
    has_contra_involvement = max_llh_data["contra"].sum(axis=1) != 0
    column_conditions = {
        ("ipsi"  , "all"  ): [True] * len(dataset),
        ("ipsi"  , "T1/T2"): dataset["tumor", "1", "t_stage"].isin([1, 2]),
        ("ipsi"  , "T3/T4"): dataset["tumor", "1", "t_stage"].isin([3, 4]),
        ("ipsi"  , "ECE+" ): has_extracapsular & is_nplus,
        ("ipsi"  , "ECE-" ): has_no_extracapsular & is_nplus,
        ("contra", "all"  ): [True] * len(dataset),
    }

    columns = pd.MultiIndex.from_tuples(column_conditions.keys())
    index = pd.MultiIndex.from_tuples(
        product(["pos", "neg"], repeat=3),
        names=["I", "II", "III"],
    )
    table = pd.DataFrame(index=index, columns=columns)

    for (side, col), col_condition in column_conditions.items():
        row_conditions = {}
        for lnl_I, lnl_II, lnl_III in product([True, False], repeat=3):
            row_conditions[(tf2str(lnl_I), tf2str(lnl_II), tf2str(lnl_III))] = (
                (max_llh_data[side]["I"] == lnl_I)
                & (max_llh_data[side]["II"] == lnl_II)
                & (max_llh_data[side]["III"] == lnl_III)
            )
        for row, row_condition in row_conditions.items():
            subset = max_llh_data[side].loc[col_condition & row_condition]
            table.loc[row, (side, col)] = len(subset)

    table.to_csv(TABLES_DIR / OUTPUT_NAME)
