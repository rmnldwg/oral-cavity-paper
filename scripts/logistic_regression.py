# -*- coding: utf-8 -*-
"""
Logistic regression.
"""
# pylint: disable=import-error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lyscripts.plot.utils import get_size
from shared import COLORS, DATA_DIR, FIGURES_DIR, MPLSTYLE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

plt.style.use(MPLSTYLE)

# barplot settings
WIDTH, SPACE = 0.8, 0.6
LABELS = ["[0,10[", "[10,20[", "[20,30[", "[30,40[", "[40,50[", "50+"]
WIDTHS = np.array([WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH])

# compute positions of bar centers based on WIDTHS and SPACE, such that the space
# between neighboring bars is SPACE. The first bar is centered at SPACE/2 + WIDTH/2.
POSITIONS = np.zeros_like(WIDTHS)
for i, width in enumerate(WIDTHS):
    spaces = (0.5 + i) * SPACE
    widths = sum(WIDTHS[: np.maximum(0, i)]) + width / 2
    POSITIONS[i] = spaces + widths


reg_data_raw = pd.read_csv(DATA_DIR / "lymph_nodes_invest_OC.csv", sep=";")
reg_data = reg_data_raw.iloc[:, [-6, -2, -1]].dropna()
classification_data = reg_data_raw.iloc[:, -1].dropna()

reg_data.iloc[:, 0]

# Histogram with largest lymph node on x axis and fn and tp on y axis and confusion
# matrix as a table
data = {
    "": ["Clinically Positive", "Clinically Negative"],
    "Pathology Positive": [
        sum(classification_data == "tp"),
        sum(classification_data == "fn"),
    ],
    "Pathology Negative": [
        sum(classification_data == "fp"),
        sum(classification_data == "tn"),
    ],
}

conditions = [
    reg_data.iloc[:, 0] < 10,
    (reg_data.iloc[:, 0] < 20) & (reg_data.iloc[:, 0] >= 10),
    (reg_data.iloc[:, 0] < 30) & (reg_data.iloc[:, 0] >= 20),
    (reg_data.iloc[:, 0] < 40) & (reg_data.iloc[:, 0] >= 30),
    (reg_data.iloc[:, 0] < 50) & (reg_data.iloc[:, 0] >= 40),
    reg_data.iloc[:, 0] >= 50,
]
values = ["[0,10[", "[10,20[", "[20,30[", "[30,40[", "[40,50[", "50+"]

reg_data["size_category"] = np.select(conditions, values, default="Unknown")
reg_data["size_category"] = pd.Categorical(
    reg_data["size_category"],
    ["[0,10[", "[10,20[", "[20,30[", "[30,40[", "[40,50[", "50+"],
)

hist_data1 = pd.DataFrame()
hist_data2 = pd.DataFrame()
hist_data3 = pd.DataFrame()
hist_data4 = pd.DataFrame()
hist_data5 = pd.DataFrame()
hist_data6 = pd.DataFrame()

hist_data1["0-10"] = reg_data[reg_data["size_category"] == "[0,10["].iloc[:, 2]
hist_data2["10-20"] = reg_data[reg_data["size_category"] == "[10,20["].iloc[:, 2]
hist_data3["20-30"] = reg_data[reg_data["size_category"] == "[20,30["].iloc[:, 2]
hist_data4["30-40"] = reg_data[reg_data["size_category"] == "[30,40["].iloc[:, 2]
hist_data5["40-50"] = reg_data[reg_data["size_category"] == "[40,50["].iloc[:, 2]
hist_data6["50+"] = reg_data[reg_data["size_category"] == "50+"].iloc[:, 2]

hist_data = pd.concat(
    [hist_data1, hist_data2, hist_data3, hist_data4, hist_data5, hist_data6]
)

fig = plt.figure(
    figsize=get_size(width="full", ratio="golden"), constrained_layout=True
)
ax = fig.add_subplot()

hist_tp = hist_data == "tp"
hist_fn = hist_data == "fn"

plt.bar(
    POSITIONS + SPACE / 2.0,
    hist_fn.sum(),
    label="false negative (" + str(sum(reg_data.iloc[:, 2] == "fn")) + ")",
    width=WIDTHS,
    color=COLORS["red"],
)
plt.bar(
    POSITIONS,
    hist_tp.sum(),
    label="true positive (" + str(sum(reg_data.iloc[:, 2] == "tp")) + ")",
    width=WIDTHS,
    color=COLORS["green"],
)

ax.set_xticks(POSITIONS)
ax.set_xticklabels(LABELS)
ax.grid(axis="x")
ax.set_ylabel("count")
ax.set_xlabel("largest lymph node [mm]")
ax.legend()

plt.savefig(FIGURES_DIR / "confusionmat_lymphsize_hist.png")


# Create the DataFrame
df = pd.DataFrame(data).set_index("")

# Create the table plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis("off")
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    rowLabels=df.index,
    cellLoc="center",
    loc="center",
    colWidths=[0.2, 0.2, 0.2],
)

# Save the table as a PNG
plt.savefig(FIGURES_DIR / "confusion_matrix.png")


# logistic regression
X = pd.DataFrame(reg_data.iloc[:, 0])
X = X.values.reshape((-1, 1))
y = reg_data.iloc[:, 1]

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
accuracy = logreg.score(X_test, y_test)
print(f"Accuracy of logistic regression classifier on test set: {accuracy:.2f}")

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test, y_pred))

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label=f"Logistic Regression (area = {logit_roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.savefig(FIGURES_DIR / "Log_ROC.svg")
