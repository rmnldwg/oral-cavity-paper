# -*- coding: utf-8 -*-
"""
Create a table with per-level involvement prevalence for patients with and without
extracapsular extension (ECE).
"""
# pylint: disable=import-error
# pylint: disable=singleton-comparison
from pathlib import Path

import pandas as pd
from shared import DATAFILE, TABLES_DIR, load_and_prepare_data

OUTPUT_NAME = Path(__file__).with_suffix(".csv").name
LNLS = ["I", "II", "III", "IV", "V"]

if __name__ == "__main__":
    dataset, max_llh_data = load_and_prepare_data(filepath=DATAFILE, lnls=LNLS)

    is_nplus = max_llh_data.sum(axis=1) != 0
    subset_condition = {
        "N+": True,
        "ECE+": dataset["patient", "#", "extracapsular"] == True,
        "ECE?": dataset["patient", "#", "extracapsular"].isna(),
        "ECE-": dataset["patient", "#", "extracapsular"] == False,
    }

    columns = ["total", *LNLS]
    index = pd.MultiIndex.from_product(
        [["ipsi", "contra"], ["N+", "ECE+", "ECE?", "ECE-"]]
    )
    table = pd.DataFrame(index=index, columns=columns)

    for side in ["ipsi", "contra"]:
        for row in ["N+", "ECE+", "ECE?", "ECE-"]:
            subset = max_llh_data[side].loc[is_nplus & subset_condition[row]]
            table.loc[(side, row), "total"] = len(subset)

            for lnl in LNLS:
                table.loc[(side, row), lnl] = int(subset[lnl].sum())

    table.to_csv(TABLES_DIR / OUTPUT_NAME)
