# pylint: disable=singleton-comparison
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colormaps

from lyscripts.plot.histograms import get_size


OUTPUT_NAME = Path(__file__).with_suffix(".png").name
OUTPUT_DIR = Path("./figures")
MPLSTYLE = Path("./scripts/.mplstyle")

# barplot settings
WIDTH, SPACE = 0.8, 0.6
LABELS  = ["Ia","Ib ipsi","Ib contra", "II ipsi", "II contra", "III ipsi", "III contra", "IV ipsi", "IV contra", "V ipsi", "V contra"]
WIDTHS  = np.array([WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH])

# compute positions of bar centers based on WIDTHS and SPACE, such that the space
# between neighboring bars is SPACE. The first bar is centered at SPACE/2 + WIDTH/2.
POSITIONS = np.zeros_like(WIDTHS)
for i, width in enumerate(WIDTHS):
    spaces = (0.5 + i) * SPACE
    widths = sum(WIDTHS[:np.maximum(0,i)]) + width/2
    POSITIONS[i] = spaces + widths

COLORS = {
    "green": '#00afa5',
    "red": '#ae0060',
    "blue": '#005ea8',
    "orange": '#f17900',
    "gray": '#c5d5db',
}


import matplotlib.pyplot as plt
import numpy as np

data_raw = pd.read_csv("./data/lymph_nodes_invest_patincluded.csv", sep=";")

plot_data_pos = data_raw.iloc[:,[8,10,12,14,16,18,20,22,24,26,28]]
plot_data_inv = data_raw.iloc[:,[7,9,11,13,15,17,19,21,23,25,27]]

plt.style.use(MPLSTYLE)

"""
a = plt.bar(POSITIONS + SPACE/3, plot_data_inv.sum(), color="#00afa5", width=WIDTHS, label="investigated")
b = plt.bar(POSITIONS, plot_data_pos.sum(), color="#ae0060", width=WIDTHS, label="positiv")
plt.xticks(POSITIONS, LABELS)
plt.legend()$
"""

fig = plt.figure()
ax1 = fig.add_subplot(111)
  # instantiate a second axes that shares the same x-axis
a = ax1.bar(POSITIONS + SPACE/3, plot_data_inv.sum(), color="#00afa5", width=WIDTHS, label="lymph nodes investigated")

ax2 = ax1.twinx()

b = ax2.bar(POSITIONS, 100*pd.array(plot_data_pos.sum())/pd.array(plot_data_inv.sum()), color="#ae0060", width=WIDTHS, label="lymph nodes involved")


ax1.set_xticks(POSITIONS, LABELS)
ax1.set_xlabel("lymph node level")
ax1.set_ylim(0, 4000)
ax1.set_yticks(np.arange(0, 4500, step=500))
ax1.set_ylabel("number of investigated lymph nodes", color="#00afa5")
ax1.xaxis.grid(False)
ax2.set_ylabel("positiv lymph nodes [%]", color="#ae0060")
ax2.set_yticks(np.arange(0, 9, step=1))

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

plt.savefig(OUTPUT_DIR / "lymph_invest_hist.png")


#2D histogram with number of positive and investigated lymph 
colnames = ["Ia", "Ib ipsi", "Ib contra", "II ipsi", "II contra", "III ipsi", "III contra", "IV ipsi", "IV contra", "V ipsi", "V contra"]

for r in range(11):
  data = data_raw.iloc[:,[7+2*r,8+2*r]].dropna()
  colname = colnames[r]
  fig, ax = plt.subplots()

    # instantiate a second axes that shares the same x-axis
  hist = ax.hist2d(data.iloc[:,0], data.iloc[:,1], range=[(min(data.iloc[:,0])-0.5,max(data.iloc[:,0])+0.5),(min(data.iloc[:,1])-0.5,max(data.iloc[:,1])+0.5)], bins=(1+int(max(data.iloc[:,0])-min(data.iloc[:,0])), 1+int(max(data.iloc[:,1])-min(data.iloc[:,1]))), cmap=colormaps['Reds'])
  
  if max(data.iloc[:,0])<11:
    plt.xticks(np.arange(min(data.iloc[:,0]), 1+max(data.iloc[:,0])))
    plt.yticks(np.arange(min(data.iloc[:,1]), 1+max(data.iloc[:,1])))

  if max(data.iloc[:,0])>=11:
    plt.xticks(np.arange(min(data.iloc[:,0]), 1+max(data.iloc[:,0]), 2))
    plt.yticks(np.arange(min(data.iloc[:,1]), 1+max(data.iloc[:,1]), 2))
    
  plt.xlabel("number of lymph nodes investigated")
  plt.ylabel("number of positive lymph nodes")

  for i in range(len(hist[0])):
      for j in range(len(hist[0][i])):
          bin_val = hist[0][i][j]
          if bin_val > 0:
              ax.text(hist[1][i+1] - 0.5, hist[2][j+1] - 0.5, int(bin_val),
                      ha='center', va='center', color='black', fontsize=6)

  plt.savefig("./figures/lymph_invest_hist2d" + colname + ".png")
