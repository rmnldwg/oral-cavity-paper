# -*- coding: utf-8 -*-
"""
Ipsilateral and contralateral involvement stratified according to primary tumor subsite.
"""
# pylint: disable=import-error
# pylint: disable=singleton-comparison
from pathlib import Path

import pandas as pd
from shared import DATAFILE, TABLES_DIR, load_and_prepare_data

OUTPUT_NAME = Path(__file__).with_suffix(".csv").name
LNLS = ["I", "II", "III", "IV", "V"]
SUBSITES = {
    "tongue": ["C02", "C02.0", "C02.1", "C02.2", "C02.3", "C02.4", "C02.8", "C02.9"],
    "gums & cheek": ["C03", "C03.0", "C03.1", "C03.9", "C06", "C06.0", "C06.1", "C06.2", "C06.8", "C06.9"],
    "floor of mouth": ["C04", "C04.0", "C04.1", "C04.8", "C04.9"],
}

if __name__ == "__main__":
    dataset, max_llh_data = load_and_prepare_data(filepath=DATAFILE, lnls=LNLS)

    is_nplus = max_llh_data.sum(axis=1) != 0
    subset_condition = {
        "all": [True] * len(dataset),
        "tongue": dataset["tumor", "1", "subsite"].isin(SUBSITES["tongue"]),
        "gums & cheek": dataset["tumor", "1", "subsite"].isin(SUBSITES["gums & cheek"]),
        "floor of mouth": dataset["tumor", "1", "subsite"].isin(SUBSITES["floor of mouth"]),
    }

    columns = ["total", *LNLS]
    index = pd.MultiIndex.from_product(
        [["ipsi", "contra"], list(subset_condition.keys())]
    )
    table = pd.DataFrame(index=index, columns=columns)

    for side in ["ipsi", "contra"]:
        for row, condition in subset_condition.items():
            subset = max_llh_data[side].loc[condition]
            table.loc[(side, row), "total"] = len(subset)

            for lnl in LNLS:
                table.loc[(side, row), lnl] = int(subset[lnl].sum())

    table.to_csv(TABLES_DIR / OUTPUT_NAME)
