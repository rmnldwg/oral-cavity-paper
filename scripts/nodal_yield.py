# -*- coding: utf-8 -*-
"""Compute how many patients had a nodal yield below 20 LNs during neck dissection."""

import pandas as pd


def main() -> None:
    """Run main."""
    data = pd.read_csv("data/enhanced.csv", header=[0, 1, 2])
    nodal_yield = data["total_dissected", "info", "all_lnls"]
    num_low_yield = nodal_yield[nodal_yield < 20].count()
    print(f"Number of patients with nodal yield < 20: {num_low_yield}")
    print(f"Median nodal yield: {nodal_yield.median()}")


if __name__ == "__main__":
    main()
