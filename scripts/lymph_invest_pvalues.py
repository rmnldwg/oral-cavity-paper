# -*- coding: utf-8 -*-
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# proportions_ztest ipsilateral
# level I
count = np.array([28, 43])
nobs = np.array([225, 123])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")
print("p-value lv I = {0:0.6f}".format(pval))

# level II
count = np.array([51, 47])
nobs = np.array([225, 123])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")
print("p-value lv II = {0:0.6f}".format(pval))

# level III
count = np.array([25, 18])
nobs = np.array([225, 123])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")
print("p-value lv III = {0:0.6f}".format(pval))

# level I, depending on involvement of level II
count = np.array([32, 39])
nobs = np.array([98, 250])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")
print("p-value lv I, depending on lv II involvement = {0:0.6f}".format(pval))

# level III, depending on involvement of level II
count = np.array([25, 18])
nobs = np.array([98, 250])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")
print("p-value lv III, depending on lv II involvement = {0:0.6f}".format(pval))

# proportions_ztest contralateral
# level I, no midext compared to midline extension
count = np.array([7, 6])
nobs = np.array([169, 33])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")
print("p-value lv I, depending on midext = {0:0.6f}".format(pval))


# level II, no midext compared to midline extension
count = np.array([4, 2])
nobs = np.array([169, 33])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")
print("p-value lv II, depending on midext = {0:0.6f}".format(pval))

# level III, no midext compared to midline extension
count = np.array([3, 1])
nobs = np.array([169, 33])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")
print("p-value lv III, depending on midext = {0:0.6f}".format(pval))

# proportions_ztest contralateral
# level I, ipsi II healthy vs involved
count = np.array([18, 4])
nobs = np.array([250, 98])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")
print("p-value lv I, depending on ipsi lv2 = {0:0.6f}".format(pval))


# to small population
# level II, ipsi II healthy vs involved
count = np.array([7, 7])
nobs = np.array([250, 98])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")
print("p-value lv II, depending on ipsi lv2 = {0:0.6f}".format(pval))

# level III, ipsi II healthy vs involved
count = np.array([4, 6])
nobs = np.array([250, 98])
stat, pval = proportions_ztest(count, nobs, alternative="two-sided")
print("p-value lv III, depending on ipsi lv2 = {0:0.6f}".format(pval))
