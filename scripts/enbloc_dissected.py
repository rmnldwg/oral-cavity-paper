# -*- coding: utf-8 -*-
"""Compute how many patients had enbloc dissections."""

import numpy as np
import pandas as pd


def together_resected(row: pd.Series) -> list[pd.Series]:
    """Determine which LNLs where resected together.

    Every patient may have multiple enbloc resections, where multiple LNLs are removed
    and sent to the pathologist together. This is indicated by the `("enbloc_dissected",
    <"ipsi"/"contra">, <LNL>)` columns. It contains a number and a letter. So, e.g., if
    LNL II and III both contain the number `13a`, then 13 LNs have been found in those
    two LNLs.

    Since in one patient, many enbloc resections may have been performed, this function
    returns a list of Series, where each Series contains boolean values indicating
    whether the LNLs were resected together.

    >>> row = pd.Series({
    ...     "I": "",
    ...     "II": "13a",
    ...     "III": "13a",
    ...     "IV": "",
    ... })
    >>> together_resected(row)   # doctest: +NORMALIZE_WHITESPACE
    [I      False
     II     True
     III    True
     IV     False
     dtype: bool]
    >>> row = pd.Series({
    ...     "I": "",
    ...     "II": "13a",
    ...     "III": "13a+9b",
    ...     "IV": "9b",
    ... })
    >>> together_resected(row)   # doctest: +NORMALIZE_WHITESPACE
    [I      False
     II     True
     III    True
     IV     False
     dtype: bool,
     I      False
     II     False
     III    True
     IV     True
     dtype: bool]
    """
    result = []

    for char in "abcdefgh":
        tmp = row.str.contains(char, na=False)
        if tmp.any():
            result.append(tmp)

    return result


def is_sublvl_pattern(row: pd.Series) -> bool:
    """Determine whether the LNLs have a sublevel pattern.

    Some LNLs are subdivided into sublevels a and b. If these are resected together,
    it is not much of an issue. So, for example if a `row` pattern is provided from
    the output of the `together_resected` function, this function will determine
    if it was only e.g. LNLs IIa and IIb that were resected together, or if it was
    across different LNLs.

    If only LNL Ia was "resected together" that probably means that it was resected
    together with the contralateral LNL Ia, as they are hard to separate. We also
    considered this as a sublevel pattern.

    >>> row = pd.Series({
    ...     "I": False,
    ...     "IIa": True,
    ...     "IIb": True,
    ...     "III": False,
    ... })
    >>> is_sublvl_pattern(row)
    True
    >>> row = pd.Series({
    ...     "I": True,
    ...     "IIa": True,
    ...     "IIb": False,
    ...     "III": False,
    ... })
    >>> is_sublvl_pattern(row)
    False
    """
    lnls_with_sublvls = set()
    for lnl in row.index:
        if "a" in lnl or "b" in lnl:
            lnls_with_sublvls.add(lnl[:-1])

    sublvl_patterns = []
    for lnl in lnls_with_sublvls:
        sublvl_patterns.append(row.index.str.match(rf"{lnl}[ab]"))

    for pattern in sublvl_patterns:
        if (row == pattern).all():
            return True

    return False


def main() -> None:
    """Run main."""
    data = pd.read_csv("data/enhanced.csv", header=[0, 1, 2])
    has_enbloc = pd.Series([False] * len(data), index=data.index, dtype=bool)
    ipsi_contra_Ia_count = 0
    sublvl_count = 0
    sublvl_patient_count = 0
    pattern_counts = {}

    for i, row in data["enbloc_dissected"].iterrows():
        has_ipsi_contra_Ia = False
        has_sublvl = False
        for side in ["ipsi", "contra"]:
            enbloc_patterns = together_resected(row[side])
            if len(enbloc_patterns) == 0:
                continue

            pattern_Ia = row[side].index.str.match("Ia")

            for pattern in enbloc_patterns:
                if (pattern == pattern_Ia).all():
                    has_ipsi_contra_Ia = True
                    continue

                if not is_sublvl_pattern(pattern):
                    pattern_str = "+".join(row[side].index[pattern].to_list())
                    if pattern_str in pattern_counts:
                        pattern_counts[pattern_str] += 1
                    else:
                        pattern_counts[pattern_str] = 1
                    has_enbloc[i] = True
                else:
                    has_sublvl = True
                    sublvl_count += 1

        ipsi_contra_Ia_count += has_ipsi_contra_Ia
        sublvl_patient_count += has_sublvl

    pattern_strs = list(pattern_counts.keys())
    pattern_counts = list(pattern_counts.values())
    sort_idx = np.argsort(pattern_counts)[::-1]
    sorted_pattern_strs = [pattern_strs[i] for i in sort_idx]
    sorted_pattern_counts = [pattern_counts[i] for i in sort_idx]

    num_enbloc = has_enbloc.sum()
    print(
        f"Patients with enbloc dissection of ipsi and contra Ia: {ipsi_contra_Ia_count}"
    )
    print(
        f"Co-resected sublevels (e.g. IIa and IIb): {sublvl_count} in {sublvl_patient_count} patients"
    )
    print(f"Patients with some other type of enbloc dissection: {num_enbloc}")
    print(
        "Most common three patterns: "
        + ", ".join(
            [
                f"{p} ({c})"
                for p, c in zip(
                    sorted_pattern_strs[:3],
                    sorted_pattern_counts[:3],
                )
            ]
        )
    )


if __name__ == "__main__":
    main()
