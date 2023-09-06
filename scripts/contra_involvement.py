# -*- coding: utf-8 -*-
"""
Look at correlations between involvment in the lymph node levels I, II, and III.
"""
# pylint: disable=import-error
# pylint: disable=singleton-comparison
from itertools import product
from pathlib import Path

import pandas as pd
from shared import DATAFILE, TABLES_DIR, add_percent, load_and_prepare_data

OUTPUT_NAME = Path(__file__).with_suffix(".csv").name
LNLS = ["I", "II", "III", "IV", "V"]


if __name__ == "__main__":
    dataset, max_llh_data = load_and_prepare_data(filepath=DATAFILE, lnls=LNLS)

    t_stage = {
        "early": dataset["tumor", "1", "t_stage"] <= 2,
        "late": dataset["tumor", "1", "t_stage"] > 2,
    }

    midline = {
        "yes": dataset["tumor", "1", "extension"] == True,
        "no": dataset["tumor", "1", "extension"] == False,
        "unknown": dataset["tumor", "1", "extension"].isna(),
    }

    ipsi = {
        "none": max_llh_data["ipsi"].sum(axis=1) == 0,
        "one LNL": max_llh_data["ipsi"].sum(axis=1) == 1,
        "≥ two LNLs": max_llh_data["ipsi"].sum(axis=1) > 1,
    }

    key_product = list(product(t_stage.keys(), midline.keys(), ipsi.keys()))
    val_product = list(product(t_stage.values(), midline.values(), ipsi.values()))

    columns = ["total", *add_percent(LNLS)]
    index = pd.MultiIndex.from_tuples(key_product, names=["t_stage", "midline", "ipsi"])
    data = pd.DataFrame(index=index, columns=columns)

    for key_tuple, (t, m, i) in zip(key_product, val_product):
        subset = max_llh_data.loc[t & m & i]
        data.loc[key_tuple, "total"] = len(subset)
        for lnl in LNLS:
            num = subset["contra", lnl].sum()
            data.loc[key_tuple, lnl] = num
            data.loc[key_tuple, lnl + "%"] = round(num / len(subset) * 100)

    data.to_csv(TABLES_DIR / OUTPUT_NAME)
