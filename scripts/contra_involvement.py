"""
Look at correlations between involvment in the lymph node levels I, II, and III.
"""
# pylint: disable=singleton-comparison
from itertools import product
from pathlib import Path

import pandas as pd

from statistics import ORAL_CAVITY_ICD_CODES, DATAFILE


OUTPUT_DIR = Path("tables")
OUTPUT_NAME = Path(__file__).with_suffix(".csv").name

LNLS = ["I", "II", "III", "IV", "V"]


if __name__ == "__main__":
    dataset = pd.read_csv(DATAFILE, header=[0,1,2])
    is_oral_cavity = dataset["tumor", "1", "subsite"].isin(
        icd for icd_list in ORAL_CAVITY_ICD_CODES.values() for icd in icd_list
    )
    dataset = dataset.loc[is_oral_cavity]
    max_llh_data = dataset["max_llh"]

    cols_to_drop = []
    for lnl in max_llh_data.columns.get_level_values(1):
        if lnl not in LNLS:
            cols_to_drop.append(("ipsi", lnl))
            cols_to_drop.append(("contra", lnl))

    max_llh_data = max_llh_data.drop(columns=cols_to_drop)

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
        "one lnl": max_llh_data["ipsi"].sum(axis=1) == 1,
        "two or more lnls": max_llh_data["ipsi"].sum(axis=1) > 1,
    }

    key_product = list(product(t_stage.keys(), midline.keys(), ipsi.keys()))
    val_product = list(product(t_stage.values(), midline.values(), ipsi.values()))

    columns = ["total", *LNLS]
    index = pd.MultiIndex.from_tuples(key_product, names=["t_stage", "midline", "ipsi"])
    data = pd.DataFrame(index=index, columns=columns)

    for key_tuple, (t,m,i) in zip(key_product, val_product):
        subset = max_llh_data.loc[t & m & i]
        data.loc[key_tuple, "total"] = len(subset)
        for lnl in LNLS:
            data.loc[key_tuple, lnl] = subset["contra", lnl].sum()

    data.to_csv(OUTPUT_DIR / OUTPUT_NAME)
