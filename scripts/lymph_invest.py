# -*- coding: utf-8 -*-
# pylint: disable=singleton-comparison
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lyscripts.plot.histograms import get_size
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread
from shared import DATAFILE, MPLSTYLE, load_and_prepare_data

# OUTPUT_NAME = Path(__file__).with_suffix(".png").name
OUTPUT_DIR = Path("./figures")
MPLSTYLE = Path("./scripts/.mplstyle")

# barplot settings
WIDTH, SPACE = 0.8, 0.6
LABELS = ["Ib ipsi", "II ipsi", "III ipsi", "IV ipsi", "V ipsi"]
WIDTHS = np.array([WIDTH, WIDTH, WIDTH, WIDTH, WIDTH])

# compute positions of bar centers based on WIDTHS and SPACE, such that the space
# between neighboring bars is SPACE. The first bar is centered at SPACE/2 + WIDTH/2.
POSITIONS = np.zeros_like(WIDTHS)
for i, width in enumerate(WIDTHS):
    spaces = (0.5 + i) * SPACE
    widths = sum(WIDTHS[: np.maximum(0, i)]) + width / 2
    POSITIONS[i] = spaces + widths

# USZ colors
usz_blue = "#005ea8"
usz_green = "#00afa5"
usz_red = "#ae0060"
usz_orange = "#f17900"
usz_gray = "#c5d5db"

# colormaps
white_to_blue = LinearSegmentedColormap.from_list(
    "white_to_blue", ["#ffffff", usz_blue], N=256
)
white_to_green = LinearSegmentedColormap.from_list(
    "white_to_green", ["#ffffff", usz_green], N=256
)
green_to_red = LinearSegmentedColormap.from_list(
    "green_to_red", [usz_green, usz_red], N=256
)

# Create a custom colormap. Assign white to 0
colors = ["white", usz_green, usz_red]
positions = [0, 0.001, 1]
green_to_red_modified = LinearSegmentedColormap.from_list(
    "green_to_red_modified", list(zip(positions, colors)), N=256
)


h = usz_gray.lstrip("#")
gray_rgba = tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)) + (1.0,)
tmp = LinearSegmentedColormap.from_list("tmp", [usz_green, usz_red], N=128)
tmp = tmp(np.linspace(0.0, 1.0, 128))
tmp = np.vstack([np.array([gray_rgba] * 128), tmp])
halfGray_halfGreenToRed = ListedColormap(tmp)

data_raw, _ = load_and_prepare_data(filepath=DATAFILE, lnls=["I", "II", "III"])

header1_pos = "positive_dissected"
header1_inv = "total_dissected"
header2 = "ipsi"
header3 = ["Ib", "II", "III", "IV", "V"]

plot_data_inv = data_raw.loc[:, (header1_inv, header2, header3)]
plot_data_pos = data_raw.loc[:, (header1_pos, header2, header3)]

plt.style.use(MPLSTYLE)

"""
a = plt.bar(POSITIONS + SPACE/3, plot_data_inv.sum(), color="#00afa5", width=WIDTHS, label="investigated")
b = plt.bar(POSITIONS, plot_data_pos.sum(), color="#ae0060", width=WIDTHS, label="positiv")
plt.xticks(POSITIONS, LABELS)
plt.legend()$
"""

fig = plt.figure(figsize=get_size(width="full", ratio=1.8), constrained_layout=True)
ax1 = fig.add_subplot(111)
# instantiate a second axes that shares the same x-axis
a = ax1.bar(
    POSITIONS + SPACE / 3,
    plot_data_inv.sum(),
    color="#00afa5",
    width=WIDTHS,
    label="lymph nodes investigated",
)

ax2 = ax1.twinx()

b = ax2.bar(
    POSITIONS,
    100 * pd.array(plot_data_pos.sum()) / pd.array(plot_data_inv.sum()),
    color="#ae0060",
    width=WIDTHS,
    label="lymph nodes involved",
)


ax1.set_xticks(POSITIONS, LABELS)
ax1.set_xlabel("lymph node level")
ax1.set_ylim(0, 4000)
ax1.set_yticks(np.arange(0, 8800, step=800))
ax1.set_ylabel("number of investigated lymph nodes", color="#00afa5")
ax1.xaxis.grid(False)
ax2.set_ylabel("positiv lymph nodes [%]", color="#ae0060")
ax2.set_yticks(np.arange(0, 11, step=1))

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.title("Number of lymph nodes investigated/involved per level")

plt.savefig("./figures/lymph_invest_hist_OC.png")


data_raw[("total_dissected", "ipsi + contra", "Ia")] = (
    data_raw[("total_dissected", "ipsi", "Ia")]
    + data_raw[("total_dissected", "contra", "Ia")]
)
data_raw[("positive_dissected", "ipsi + contra", "Ia")] = (
    data_raw[("positive_dissected", "ipsi", "Ia")]
    + data_raw[("positive_dissected", "contra", "Ia")]
)

# 2D histogram with number of positive and investigated lymph
colnames = [
    "Ia ipsi",
    "Ia",
    "Ib ipsi",
    "Ib contra",
    "II ipsi",
    "II contra",
    "III ipsi",
    "III contra",
    "IV ipsi",
    "IV contra",
    "V ipsi",
    "V contra",
    "Ib-III ipsi",
    "Ib-III contra",
]

levels = [
    "Ia",
    "Ia",
    "Ib",
    "Ib",
    "II",
    "II",
    "III",
    "III",
    "IV",
    "IV",
    "V",
    "V",
    "Ib_to_III",
    "Ib_to_III",
]

sides = [
    "ipsi",
    "ipsi + contra",
    "ipsi",
    "contra",
    "ipsi",
    "contra",
    "ipsi",
    "contra",
    "ipsi",
    "contra",
    "ipsi",
    "contra",
    "ipsi",
    "contra",
]
for r in range(14):
    # data = data_raw.iloc[:, [2 + 2 * r, 3 + 2 * r]].dropna()

    header1 = ["total_dissected", "positive_dissected"]
    header2 = sides[r]
    header3 = levels[r]

    data = data_raw.loc[:, (header1, header2, header3)].dropna()
    colname = colnames[r]

    fig = plt.figure(figsize=get_size(width="full", ratio=1.8), constrained_layout=True)
    spec = GridSpec(
        ncols=2,
        nrows=2,
        figure=fig,
        width_ratios=[1.0, 0.16],
        height_ratios=[1.0, 0.37],
    )

    ax = fig.add_subplot(spec[0, 0])

    # instantiate a second axes that shares the same x-axis
    if levels[r] == "Ib_to_III":
        hist = plt.hist2d(
            data.iloc[:, 0],
            data.iloc[:, 1],
            range=[(-0.5, 70.5), (-0.5, 10.5)],
            bins=(71, 11),
            cmap=green_to_red_modified,
        )
        plt.yticks(np.arange(0, 11, 1))

    else:
        hist = plt.hist2d(
            data.iloc[:, 0],
            data.iloc[:, 1],
            range=[(-0.5, 42.5), (-0.5, 7.5)],
            bins=(43, 8),
            cmap=green_to_red_modified,
        )
        plt.yticks(np.arange(0, 8, 1))

    plt.grid(False)
    plt.title(colname + " (n=" + str(len(data)) + ")")
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    ax.set_ylabel("number of positive lymph nodes")

    for i in range(len(hist[0])):
        for j in range(len(hist[0][i])):
            bin_val = hist[0][i][j]
            if bin_val > 0:
                ax.text(
                    hist[1][i + 1] - 0.5,
                    hist[2][j + 1] - 0.5,
                    round(bin_val),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize="x-small",
                )

    if levels[r] == "Ib_to_III":
        ax2 = fig.add_subplot(spec[0, 1], sharey=ax)
        ax2.hist(
            data.iloc[:, 1],
            orientation="horizontal",
            bins=11,
            range=[-0.5, 10.5],
            color="#c5d5db",
        )

    else:
        ax2 = fig.add_subplot(spec[0, 1], sharey=ax)
        ax2.hist(
            data.iloc[:, 1],
            orientation="horizontal",
            bins=8,
            range=[-0.5, 7.5],
            color="#c5d5db",
        )

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.xlim(0, 300)
    plt.axhline(data.iloc[:, 1].mean(), color="k", linestyle="dashed", linewidth=0.5)
    ax2.text(
        plt.xlim()[1] / 2,
        data.iloc[:, 1].mean(),
        "mean",
        rotation="horizontal",
        rotation_mode="anchor",
        va="bottom",
        ha="center",
        fontsize="xx-small",
    )
    ax2.xaxis.set_ticks_position("top")

    def xtick_formatter(x, pos):
        if x == 0:
            return ""
        else:
            return str(int(x))

    ax2.xaxis.set_major_formatter(plt.FuncFormatter(xtick_formatter))

    if levels[r] == "Ib_to_III":
        ax3 = fig.add_subplot(spec[1, 0], sharex=ax)
        b = ax3.hist(data.iloc[:, 0], bins=71, range=[-0.5, 70.5], color="#c5d5db")

    else:
        ax3 = fig.add_subplot(spec[1, 0], sharex=ax)
        b = ax3.hist(data.iloc[:, 0], bins=43, range=[-0.5, 42.5], color="#c5d5db")

    plt.setp(ax3.get_xticklabels(), visible=True)
    plt.ylim(0, 65)
    ax3.axvline(data.iloc[:, 0].mean(), color="k", linestyle="dashed", linewidth=0.5)
    ax3.text(
        data.iloc[:, 0].mean(),
        plt.ylim()[1] / 2,
        "mean",
        rotation="vertical",
        rotation_mode="anchor",
        va="bottom",
        ha="center",
        fontsize="xx-small",
    )
    ax3.set_xlabel("number of lymph nodes investigated")

    def ytick_formatter(y, pos):
        if y == 0:
            return ""
        else:
            return str(int(y))

    ax3.yaxis.set_major_formatter(plt.FuncFormatter(ytick_formatter))

    plt.savefig("./figures/lymph_invest_hist2d" + colname + "_OC.png")

# normalized 2D histogram with number of positive and investigated lymph
for r in range(14):
    # data = data_raw.iloc[:, [2 + 2 * r, 3 + 2 * r]].dropna()

    header1 = ["total_dissected", "positive_dissected"]
    header2 = sides[r]
    header3 = levels[r]
    data = data_raw.loc[:, (header1, header2, header3)].dropna()
    colname = colnames[r]

    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    mask = x > 0
    x1 = np.array(x)[mask]
    x1 = x1.reshape(-1, 1)

    print("\n\n", header2 + header3, "\n")
    x1 = sm.add_constant(x1, prepend=0.1, has_constant="add")
    y1 = [1 if num >= 1 else 0 for num in y]
    y1 = np.array(y1)[mask]

    # linear regression
    lm = sm.OLS(y1, x1).fit()  # fitting the model
    intercept, slope = lm.params[0], lm.params[1]
    intercept_stderr, slope_stderr = lm.bse[0], lm.bse[1]
    print(lm.summary())

    x_values = np.linspace(np.min(x1[:, 1]), np.max(x1[:, 1]), len(y1))
    y_values = intercept + slope * x_values
    # Confidence interval
    predictions = lm.get_prediction(x1)
    ci = predictions.conf_int()

    # Calculate regression lines based on errors
    t_value = 1.96  # Critical value for a 95% confidence interval (adjust as needed)
    intercept_lower = intercept - t_value * intercept_stderr
    intercept_upper = intercept + t_value * intercept_stderr
    slope_lower = slope - t_value * slope_stderr
    slope_upper = slope + t_value * slope_stderr
    y_lower = intercept_lower + slope_lower * x_values
    y_upper = intercept_upper + slope_upper * x_values

    # Plotting
    fig, ax = plt.subplots()
    for i in range(len(y1)):
        plt.scatter(
            x1[i, 1], y1[i], s=sum(x1[:, 1] == x1[i, 1]), color="blue"
        )  # Individual scatter plot for each data point
    plt.plot(
        x_values, y_values, color="red", label="Linear Regression (binary variable)"
    )
    plt.plot(x_values, ci[:, 0], color="orange", linestyle="--", label="Lower CI")
    plt.plot(x_values, ci[:, 1], color="orange", linestyle="--", label="Upper CI")
    plt.fill_between(x_values, ci[:, 0], ci[:, 1], color="orange", alpha=0.2)
    plt.plot(
        x_values,
        y_lower,
        color="green",
        linestyle="-.",
        label="Regression Line (Lower Bound)",
    )
    plt.plot(
        x_values,
        y_upper,
        color="blue",
        linestyle="-.",
        label="Regression Line (Upper Bound)",
    )
    plt.xlabel("Number if lymph nodes investigated")
    plt.ylabel("Probability of involvement")
    plt.title(colname + " (n=" + str(len(data[data.iloc[:, 0] > 0])) + ")")
    plt.legend(loc="upper left")

    # Adding the formula
    formula = f"y = {intercept:.2f} +/- {intercept_stderr:.2f} + ({slope:.4f} +/- {slope_stderr:.4f}) * x"
    plt.text(
        np.min(x1[:, 1]) + np.max(x1) / 20,
        np.max(y1) - 0.2,
        formula,
        ha="left",
        va="top",
    )  # Adjust the y coordinate by subtracting 0.05

    plt.savefig("./figures/lymph_invest_lr" + colname + "_OC.png")

    fig = plt.figure(figsize=get_size(width="full", ratio=1.8), constrained_layout=True)
    spec = GridSpec(
        ncols=2,
        nrows=2,
        figure=fig,
        width_ratios=[1.0, 0.16],
        height_ratios=[1.0, 0.37],
    )

    ax = fig.add_subplot(spec[0, 0])
    if levels[r] == "Ib_to_III":
        combinations = np.zeros((71, 11))
        for i in range(combinations.shape[0]):
            for j in range(combinations.shape[1]):
                combinations[i, j] = len(
                    data[(data.iloc[:, 0] == i) & (data.iloc[:, 1] == j)]
                )

    else:
        combinations = np.zeros((43, 8))
        for i in range(combinations.shape[0]):
            for j in range(combinations.shape[1]):
                combinations[i, j] = len(
                    data[(data.iloc[:, 0] == i) & (data.iloc[:, 1] == j)]
                )

    histmatrixnorm = combinations
    for i in range(combinations.shape[0]):
        if sum(histmatrixnorm[i, :]) > 0:
            histmatrixnorm[i, :] = (
                histmatrixnorm[i, :] / sum(histmatrixnorm[i, :]) * 100
            )
        else:
            histmatrixnorm[i, :] = 0

    x_indices, y_indices = np.indices(histmatrixnorm.shape)

    # instantiate a second axes that shares the same x-axis
    if levels[r] == "Ib_to_III":
        hist = plt.hist2d(
            x_indices.flatten(),
            y_indices.flatten(),
            weights=histmatrixnorm.flatten(),
            range=[(-0.5, 70.5), (-0.5, 10.5)],
            bins=(71, 11),
            cmap=green_to_red_modified,
        )
        plt.yticks(np.arange(0, 11, 1))

    else:
        hist = plt.hist2d(
            x_indices.flatten(),
            y_indices.flatten(),
            weights=histmatrixnorm.flatten(),
            range=[(-0.5, 42.5), (-0.5, 7.5)],
            bins=(43, 8),
            cmap=green_to_red_modified,
        )
        plt.yticks(np.arange(0, 8, 1))

    """
    # plt.plot(x, intercept + slope * x, color='black', linewidth=1, linestyle='--')
    sns.regplot(
        x=x,
        y=y,
        ax=ax,
        scatter=False,
        ci=95,
        color="black",
        line_kws={"linewidth": 0.5},
        #label=f"Regression Line (p-value={pval:.2f})",
    )
    ax.collections[1].set_label("95% Confidence interval")
    plt.legend(fontsize="xx-small", loc="upper right")
    """
    plt.grid(False)
    plt.xlabel("")
    plt.title(colname + " (n=" + str(len(data)) + ")")
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    ax.set_ylabel("number of positive lymph nodes")

    bin_val = np.zeros((len(hist[0]), len(hist[0][i])))
    for i in range(len(hist[0])):
        for j in range(len(hist[0][i])):
            if hist[0][i][j] > 0:
                bin_val[i][j] = hist[0][i][j]
                ax.text(
                    hist[1][i + 1] - 0.5,
                    hist[2][j + 1] - 0.5,
                    round(round(bin_val[i][j], 0)),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize="xx-small",
                    rotation="vertical",
                )

    if levels[r] == "Ib_to_III":
        ax2 = fig.add_subplot(spec[0, 1], sharey=ax)

        # Define the histogram parameters
        bins = 11
        range_values = [-0.5, 10.5]

        # Plot the histogram
        counts, bins, patches = ax2.hist(
            data.iloc[:, 1],
            orientation="horizontal",
            bins=bins,
            range=range_values,
            color="#c5d5db",
        )

        # Annotate each bin with its corresponding value
        for count, bin, patch in zip(counts, bins, patches):
            if count > 0:
                height = patch.get_height()
                ax2.text(
                    28,  # Adjust the x-coordinate of the text position
                    patch.get_y()
                    + height / 2,  # Set the y-coordinate of the text position
                    f"{int(count)}",  # Format the value as an integer
                    ha="left",
                    va="center",  # Horizontal and vertical alignment of the text
                    fontsize="xx-small",  # Adjust the font size of the text
                    color="black",  # Set the color of the text
                )

    else:
        ax2 = fig.add_subplot(spec[0, 1], sharey=ax)

        # Define the histogram parameters
        bins = 8
        range_values = [-0.5, 7.5]

        # Plot the histogram
        counts, bins, patches = ax2.hist(
            data.iloc[:, 1],
            orientation="horizontal",
            bins=bins,
            range=range_values,
            color="#c5d5db",
        )

        # Annotate each bin with its corresponding value
        for count, bin, patch in zip(counts, bins, patches):
            if count > 0:
                height = patch.get_height()
                ax2.text(
                    28,  # Adjust the x-coordinate of the text position
                    patch.get_y()
                    + height / 2,  # Set the y-coordinate of the text position
                    f"{int(count)}",  # Format the value as an integer
                    ha="left",
                    va="center",  # Horizontal and vertical alignment of the text
                    fontsize="xx-small",  # Adjust the font size of the text
                    color="black",  # Set the color of the text
                )

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.xlim(0, 300)
    plt.axhline(data.iloc[:, 1].mean(), color="k", linestyle="dashed", linewidth=0.5)
    ax2.text(
        plt.xlim()[1] / 2 + 5,
        data.iloc[:, 1].mean(),
        "mean",
        rotation="horizontal",
        rotation_mode="anchor",
        va="bottom",
        ha="center",
        fontsize="xx-small",
    )
    ax2.xaxis.set_ticks_position("top")

    def xtick_formatter(x, pos):
        if x == 0:
            return ""
        else:
            return str(int(x))

    def cbartick_formatter(x, pos):
        if x == 0:
            return ""
        else:
            return str(int(x))

    ax2.xaxis.set_major_formatter(plt.FuncFormatter(xtick_formatter))

    if levels[r] == "Ib_to_III":
        ax3 = fig.add_subplot(spec[1, 0], sharex=ax)

        # Define the histogram parameters
        bins = 71
        range_values = [-0.5, 70.5]

        # Plot the histogram and get the counts and bins
        counts, bins, patches = ax3.hist(
            data.iloc[:, 0],
            bins=bins,
            range=range_values,
            color="#c5d5db",
        )

        # Annotate each bin with its corresponding value
        for count, bin, patch in zip(counts, bins, patches):
            if count > 0:
                height = patch.get_height()
                ax3.text(
                    bin + 0.5,  # Adjust the x-coordinate of the text position
                    6,  # Set the y-coordinate of the text position
                    f"{int(count)}",  # Format the value as an integer
                    ha="center",
                    va="bottom",  # Horizontal and vertical alignment of the text
                    fontsize="xx-small",  # Adjust the font size of the text
                    color="black",  # Set the color of the text
                    rotation=90,
                )

    else:
        ax3 = fig.add_subplot(spec[1, 0], sharex=ax)

        # Define the histogram parameters
        bins = 43
        range_values = [-0.5, 42.5]

        # Plot the histogram and get the counts and bins
        counts, bins, patches = ax3.hist(
            data.iloc[:, 0],
            bins=bins,
            range=range_values,
            color="#c5d5db",
        )

        # Annotate each bin with its corresponding value
        for count, bin, patch in zip(counts, bins, patches):
            if count > 0:
                height = patch.get_height()
                ax3.text(
                    bin + 0.5,  # Adjust the x-coordinate of the text position
                    6,  # Set the y-coordinate of the text position
                    f"{int(count)}",  # Format the value as an integer
                    ha="center",
                    va="bottom",  # Horizontal and vertical alignment of the text
                    fontsize="xx-small",  # Adjust the font size of the text
                    color="black",  # Set the color of the text
                    rotation=90,
                )

    plt.setp(ax3.get_xticklabels(), visible=True)
    plt.ylim(0, 65)
    ax3.axvline(data.iloc[:, 0].mean(), color="k", linestyle="dashed", linewidth=0.5)
    ax3.text(
        data.iloc[:, 0].mean(),
        plt.ylim()[1] / 2,
        "mean",
        rotation="vertical",
        rotation_mode="anchor",
        va="bottom",
        ha="center",
        fontsize="xx-small",
    )
    ax3.set_xlabel("number of lymph nodes investigated")

    def ytick_formatter(y, pos):
        if y == 0:
            return ""
        else:
            return str(int(y))

    ax3.yaxis.set_major_formatter(plt.FuncFormatter(ytick_formatter))

    cbar = plt.colorbar(ax=ax2)
    # cbar = plt.colorbar(ticks=[0, 20, 40, 60 ,80 ,100])
    cbar.set_ticklabels(
        ["", "20", "40", "60", "80", "100"]
    )  # vertically oriented colorbar
    cbar.set_label("[%]")

    plt.savefig("./figures/lymph_invest_hist2dnorm" + colname + "_OC.png")

    plt.close()

"""
# wrong!!!: need to change data file as it does not differentiate between no information because of not resected or resected together with other levels
# influence of number of positive lymph node levels in level 2 on the percentage of involved patients in level 3
ipsiII_pos = data_raw.iloc[:, [3 + 2 * 4]].fillna(-1)
ipsiIII_involved = data_raw.iloc[:, [3 + 2 * 6]].fillna(-1)
contraIII_involved = data_raw.iloc[:, [3 + 2 * 7]].fillna(-1)
III_involved = (ipsiIII_involved["3 ipsi pos"] + contraIII_involved["3 contra pos"]) > 0
III_tot = (ipsiIII_involved["3 ipsi pos"] + contraIII_involved["3 contra pos"]) >= 0
II_III_corr = pd.concat(
    [
        pd.DataFrame(ipsiII_pos[ipsiII_pos >= 0]),
        pd.DataFrame(III_involved[III_involved >= 0], columns=["3 involved"]),
        pd.DataFrame(III_tot[III_tot >= 0], columns=["3 total"]),
    ],
    axis=1,
)
II_III_corrdata = pd.DataFrame(II_III_corr.groupby(["2 ipsi pos"]).sum())

WIDTH, SPACE = 0.8, 0.6
LABELS = ["0", "1", "2", "3", "4", "6"]
WIDTHS = np.array([WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH])

# compute positions of bar centers based on WIDTHS and SPACE, such that the space
# between neighboring bars is SPACE. The first bar is centered at SPACE/2 + WIDTH/2.
POSITIONS = np.zeros_like(WIDTHS)
for i, width in enumerate(WIDTHS):
    spaces = (0.5 + i) * SPACE
    widths = sum(WIDTHS[: np.maximum(0, i)]) + width / 2
    POSITIONS[i] = spaces + widths


fig = plt.figure(figsize=get_size(width="full", ratio=1.8), constrained_layout=True)
ax1 = fig.add_subplot(111)
# instantiate a second axes that shares the same x-axis
a = ax1.bar(
    POSITIONS,
    round(100 * II_III_corrdata["3 involved"] / II_III_corrdata["3 total"], 0),
    color=usz_red,
    width=WIDTHS,
    label="lymph nodes investigated",
)
ax1.set_xticks(POSITIONS, LABELS)
ax1.set_xlabel("lymph nodes positive level II ipsilateral")
ax1.set_ylabel("involvement level III [%]")

j = 0
for p in a.patches:
    if p.get_height() == 0:
        plt.annotate(
            str(II_III_corrdata["3 involved"][j])
            + " / "
            + str(II_III_corrdata["3 total"][j]),
            (p.get_x() + p.get_width() / 2.0, 2),
            ha="center",
            va="center",
            size=7,
            xytext=(0, 0),
            textcoords="offset points",
        )
    else:
        plt.annotate(
            str(II_III_corrdata["3 involved"][j])
            + " / "
            + str(II_III_corrdata["3 total"][j]),
            (p.get_x() + p.get_width() / 2.0, p.get_height() / 2.0),
            ha="center",
            va="center",
            size=7,
            xytext=(0, 0),
            textcoords="offset points",
        )
    if j == 4:
        j += 2
    else:
        j += 1

plt.savefig("./figures/II_III_corr_OC.png")
"""

"""
#anpassen, da in enhanced.csv nicht Info für alle lymph nodes enthalten ist
# boxplot for the number of investigated lymph nodes
fig = plt.figure(figsize=get_size(width="full", ratio=1.8), constrained_layout=True)
ax = fig.add_subplot(111)


b = ax.boxplot(data_raw["total invest"])
ax.set_ylabel("number of lymph nodes")

plt.savefig("./figures/investigated_LN_OC.png")
"""

# generate the plot with the 2d histograms as subplots
# Load the image files
image1 = imread("./figures/lymph_invest_hist2dIb ipsi_OC.png")
image2 = imread("./figures/lymph_invest_hist2dII ipsi_OC.png")
image3 = imread("./figures/lymph_invest_hist2dIII ipsi_OC.png")
image4 = imread("./figures/lymph_invest_hist2dIV ipsi_OC.png")
image5 = imread("./figures/lymph_invest_hist2dV ipsi_OC.png")
image6 = imread("./figures/lymph_invest_hist_OC.png")

image11 = imread("./figures/lymph_invest_hist2dnormIb ipsi_OC.png")
image22 = imread("./figures/lymph_invest_hist2dnormII ipsi_OC.png")
image33 = imread("./figures/lymph_invest_hist2dnormIII ipsi_OC.png")


# Create a new figure
fig = plt.figure(
    figsize=get_size(width="full", ratio=1.2), constrained_layout=True, dpi=1000
)

# Add subplots and display images
ax1 = fig.add_subplot(3, 2, 1)
ax1.imshow(image1)
ax1.axis("off")

ax2 = fig.add_subplot(3, 2, 2)
ax2.imshow(image2)
ax2.axis("off")

ax3 = fig.add_subplot(3, 2, 3)
ax3.imshow(image3)
ax3.axis("off")

ax4 = fig.add_subplot(3, 2, 4)
ax4.imshow(image4)
ax4.axis("off")

ax5 = fig.add_subplot(3, 2, 5)
ax5.imshow(image5)
ax5.axis("off")

ax6 = fig.add_subplot(3, 2, 6)
ax6.imshow(image6)
ax6.axis("off")

# Show the figure
plt.savefig("./figures/lymph_invest_hist2ds_combined_OC.png")

fig = plt.figure(
    figsize=get_size(width="full", ratio=1.2), constrained_layout=True, dpi=1000
)

# Add subplots and display images
ax1 = fig.add_subplot(3, 1, 1)
ax1.imshow(image11)
ax1.axis("off")

ax2 = fig.add_subplot(3, 1, 2)
ax2.imshow(image22)
ax2.axis("off")

ax3 = fig.add_subplot(3, 1, 3)
ax3.imshow(image33)
ax3.axis("off")

# Show the figure
plt.savefig("./figures/lymph_invest_hist2dsnorm_combined_OC.png")
