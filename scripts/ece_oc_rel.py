# pylint: disable=singleton-comparison
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib import gridspec

from lyscripts.plot.histograms import get_size


OUTPUT_NAME = Path(__file__).with_suffix(".png").name
OUTPUT_DIR = Path("./figures")
MPLSTYLE = Path("./scripts/.mplstyle")


# barplot settings
WIDTH, SPACE = 0.8, 0.6
LABELS  =          ["ppp"   , "ppn"   , "pnp" , "pnn", "npp" , "npn", "nnp", "nnn", "total" ]
WIDTHS  = np.array([WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH])

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

# Groups of data, first values are plotted on top of each other
# Second values are plotted on top of each other, etc
ece = np.array([9,11,2,14,9,21,5,9,80])
noece = np.array([2,11,2,22,5,31,9,185,267])
ece_unk = np.array([0,0,0,0,0,0,0,1,1])

ece_ipsi = np.array([9,10,2,12,8,15,5,2,63])
noece_ipsi = np.array([2,11,2,22,5,31,9,190,272])
ece_ipsi_unk = np.array([0,1,0,2,1,6,0,3,13])

ece_contra = np.array([0,0,0,0,1,0,0,5,6])
noece_contra = np.array([11,21,4,34,12,46,14,187,329])
ece_contra_unk = np.array([0,1,0,2,1,6,0,3,13])

sums = ece + noece + ece_unk
sums_ipsi = ece_ipsi + noece_ipsi + ece_ipsi_unk
sums_contra = ece_contra + noece_contra + ece_contra_unk

plt.style.use(MPLSTYLE)

a = plt.bar(POSITIONS, ece/sums*100, label= "ECE (" + str(int(sum(ece)/2)) + ")", color="#ae0060", width=WIDTHS)
b = plt.bar(POSITIONS, noece/sums*100, bottom=ece/sums*100, label="no ECE ("  + str(int(sum(noece)/2)) + ")", color="#00afa5", width=WIDTHS)
c = plt.bar(POSITIONS, ece_unk/sums*100, bottom=(ece + noece)/sums*100, label="unknown ("  + str(int(sum(ece_unk)/2)) + ")", color="#c5d5db", width=WIDTHS)

i = 0
for p in a.patches:
    plt.annotate(format(p.get_height()/100*sums[i], '.0f'), 
                 (p.get_x() + p.get_width() / 2., p.get_height()/2), 
                   ha = 'center', va = 'center', 
                   size=7,
                   xytext = (0, 0), 
                   textcoords = 'offset points')
    i += 1

j = 0
for p in b.patches:
    plt.annotate(format(p.get_height()/100*sums[j], '.0f'), 
                 (p.get_x() + p.get_width() / 2., p.get_height()/2 +ece[j]/sums[j]*100), 
                   ha = 'center', va = 'center', 
                   size=7,
                   xytext = (0, 0), 
                   textcoords = 'offset points')
    j += 1

k = 0
for p in c.patches:
    if p.get_height() > 3:
        plt.annotate(format(p.get_height()/100*sums[k], '.0f'), 
                    (p.get_x() + p.get_width() / 2., 100 - 100*ece_unk[k]/sums[k]/2), 
                    ha = 'center', va = 'center', 
                    size=7,
                    xytext = (0, 0), 
                    textcoords = 'offset points')
    if p.get_height() < 3:
        if p.get_height() > 0:
            plt.annotate(format(p.get_height()/100*sums[k], '.0f'), 
                        (p.get_x() + p.get_width() / 2., 98.5-p.get_height()), 
                        ha = 'center', va = 'center', 
                        size=7,
                        xytext = (0, 0), 
                        textcoords = 'offset points')
    k += 1

plt.xticks(POSITIONS, LABELS)
plt.xlabel("ipsilateral involvement (I-III)")
plt.ylabel("[%]")
plt.legend(loc="upper right")
ax = plt.gca()
ax.xaxis.grid(False)

plt.savefig(OUTPUT_DIR / OUTPUT_NAME)