# -*- coding: utf-8 -*-
# pylint: disable=singleton-comparison
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread

from lyscripts.plot.histograms import get_size

OUTPUT_NAME = Path(__file__).with_suffix(".png").name
OUTPUT_DIR = Path("./figures")
MPLSTYLE = Path("./scripts/.mplstyle")

# barplot settings
WIDTH, SPACE = 0.8, 0.6
LABELS  = ["Ib ipsi", 
           "II ipsi", 
           "III ipsi", 
           "IV ipsi", 
           "V ipsi"]
WIDTHS  = np.array(
    [WIDTH, WIDTH, WIDTH, WIDTH, WIDTH]
    )

# compute positions of bar centers based on WIDTHS and SPACE, such that the space
# between neighboring bars is SPACE. The first bar is centered at SPACE/2 + WIDTH/2.
POSITIONS = np.zeros_like(WIDTHS)
for i, width in enumerate(WIDTHS):
    spaces = (0.5 + i) * SPACE
    widths = sum(WIDTHS[:np.maximum(0,i)]) + width/2
    POSITIONS[i] = spaces + widths

# USZ colors
usz_blue = '#005ea8'
usz_green = '#00afa5'
usz_red = '#ae0060'
usz_orange = '#f17900'
usz_gray = '#c5d5db'

# colormaps
white_to_blue  = LinearSegmentedColormap.from_list("white_to_blue", 
                                                   ["#ffffff", usz_blue], 
                                                   N=256)
white_to_green = LinearSegmentedColormap.from_list("white_to_green", 
                                                   ["#ffffff", usz_green], 
                                                   N=256)
green_to_red   = LinearSegmentedColormap.from_list("green_to_red", 
                                                   [usz_green, usz_red], 
                                                   N=256)

h = usz_gray.lstrip('#')
gray_rgba = tuple(int(h[i:i+2], 16) / 255. for i in (0, 2, 4)) + (1.0,)
tmp = LinearSegmentedColormap.from_list("tmp", [usz_green, usz_red], N=128)
tmp = tmp(np.linspace(0., 1., 128))
tmp = np.vstack([np.array([gray_rgba]*128), tmp])
halfGray_halfGreenToRed = ListedColormap(tmp)

def set_size(width="single", unit="cm", ratio="golden"):
    if width == "single":
        width = 10
    elif width == "full":
        width = 16
    else:
        try:
            width = width
        except:
            width = 10
            
    if unit == "cm":
        width = width / 2.54
        
    if ratio == "golden":
        ratio = 1.618
    else:
        ratio = ratio
    
    try:
        height = width / ratio
    except:
        height = width / 1.618
        
    return (width, height)

data_raw = pd.read_csv("./data/lymph_nodes_invest_OC.csv", sep=";")

plot_data_pos = data_raw.iloc[:,[7,11,15,19,23]]
plot_data_inv = data_raw.iloc[:,[6,10,14,18,22]]

plt.style.use(MPLSTYLE)

"""
a = plt.bar(POSITIONS + SPACE/3, plot_data_inv.sum(), color="#00afa5", width=WIDTHS, label="investigated")
b = plt.bar(POSITIONS, plot_data_pos.sum(), color="#ae0060", width=WIDTHS, label="positiv")
plt.xticks(POSITIONS, LABELS)
plt.legend()$
"""

fig = plt.figure(figsize=set_size(width="full", ratio=1.8), 
                 constrained_layout=True)
ax1 = fig.add_subplot(111)
  # instantiate a second axes that shares the same x-axis
a = ax1.bar(POSITIONS + SPACE/3, plot_data_inv.sum(), color="#00afa5", width=WIDTHS, label= "lymph nodes investigated")

ax2 = ax1.twinx()

b = ax2.bar(POSITIONS, 100*pd.array(plot_data_pos.sum())/pd.array(plot_data_inv.sum()), color="#ae0060", width=WIDTHS, label="lymph nodes involved")


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

plt.savefig(OUTPUT_DIR / "lymph_invest_hist_OC.png")


#2D histogram with number of positive and investigated lymph 
colnames = ["Ia ipsi", "Ia", "Ib ipsi", "Ib contra", "II ipsi", "II contra", "III ipsi", "III contra", "IV ipsi", "IV contra", "V ipsi", "V contra"]

for r in range(12):
  data = data_raw.iloc[:,[2+2*r,3+2*r]].dropna()
  colname = colnames[r]

  fig = plt.figure(figsize=set_size(width="full", ratio=1.8), 
                 constrained_layout=True)
  spec = GridSpec(ncols=2, nrows=2, figure=fig, 
                   width_ratios=[1., 0.16], height_ratios=[1., 0.37])
  
  ax = fig.add_subplot(spec[0,0])

    # instantiate a second axes that shares the same x-axis
  hist = plt.hist2d(data.iloc[:,0], data.iloc[:,1], range=[(-0.5,42.5),(-0.5,7.5)], bins=(43, 8), cmap=green_to_red)
  plt.title(colname + " (n=" + str(len(data)) + ")")

  plt.yticks(np.arange(0, 8, 1))
  plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
   ) # labels along the bottom edge are off
    
  ax.set_ylabel("number of positive lymph nodes")

  for i in range(len(hist[0])):
      for j in range(len(hist[0][i])):
          bin_val = hist[0][i][j]
          if bin_val > 0:
              ax.text(hist[1][i+1] - 0.5, hist[2][j+1] - 0.5, int(bin_val),
                      ha='center', va='center', color='white', fontsize='x-small')
  
  ax2 = fig.add_subplot(spec[0,1], sharey=ax)
  ax2.hist(data.iloc[:,1], orientation="horizontal", bins = 8, range=[-0.5,7.5], color="#c5d5db")
  plt.setp(ax2.get_yticklabels(), visible=False);
  plt.xlim(0,300)
  plt.axhline(data.iloc[:,1].mean(), color='k', linestyle='dashed', linewidth=0.5)
  ax2.text(plt.xlim()[1]/2, data.iloc[:,1].mean(), 'mean', rotation='horizontal', rotation_mode='anchor', va='bottom', ha='center', fontsize='xx-small')
  ax2.xaxis.set_ticks_position('top')
  def xtick_formatter(x, pos):
    if x == 0:
        return ''
    else:
        return str(int(x))
  ax2.xaxis.set_major_formatter(plt.FuncFormatter(xtick_formatter))

  ax3 = fig.add_subplot(spec[1,0], sharex=ax)
  b = ax3.hist(data.iloc[:,0], bins = 43, range=[-0.5,42.5], color="#c5d5db")
  plt.setp(ax3.get_xticklabels(), visible=True);
  plt.ylim(0,65)
  ax3.axvline(data.iloc[:,0].mean(), color='k', linestyle='dashed', linewidth=0.5)
  ax3.text(data.iloc[:,0].mean(), plt.ylim()[1]/2, 'mean', rotation='vertical', rotation_mode='anchor', va='bottom', ha='center', fontsize='xx-small')
  ax3.set_xlabel("number of lymph nodes investigated")
  def ytick_formatter(y, pos):
    if y == 0:
        return ''
    else:
        return str(int(y))
  ax3.yaxis.set_major_formatter(plt.FuncFormatter(ytick_formatter))

  plt.savefig("./figures/lymph_invest_hist2d" + colname + "_OC.png")


#influence of number of positive lymph node levels in level 2 on the percentage of involved patients in level 3
ipsiII_pos = data_raw.iloc[:,[3+2*4]].fillna(0)
ipsiIII_involved = data_raw.iloc[:,[3+2*6]].fillna(0)
contraIII_involved = data_raw.iloc[:,[3+2*7]].fillna(0)
III_involved = (ipsiIII_involved['3 ipsi pos'] + contraIII_involved['3 contra pos']) > 0
III_tot = (ipsiIII_involved['3 ipsi pos'] + contraIII_involved['3 contra pos']) >=0
II_III_corr = pd.concat([pd.DataFrame(ipsiII_pos), pd.DataFrame(III_involved, columns=['3 involved']), pd.DataFrame(III_tot, columns=['3 total'])], axis=1)
II_III_corrdata = pd.DataFrame(II_III_corr.groupby(['2 ipsi pos']).sum())

WIDTH, SPACE = 0.8, 0.6
LABELS  = ["0", "1", "2", "3", "4", "6"]
WIDTHS  = np.array([WIDTH, WIDTH, WIDTH, WIDTH, WIDTH, WIDTH])

# compute positions of bar centers based on WIDTHS and SPACE, such that the space
# between neighboring bars is SPACE. The first bar is centered at SPACE/2 + WIDTH/2.
POSITIONS = np.zeros_like(WIDTHS)
for i, width in enumerate(WIDTHS):
    spaces = (0.5 + i) * SPACE
    widths = sum(WIDTHS[:np.maximum(0,i)]) + width/2
    POSITIONS[i] = spaces + widths


fig = plt.figure(figsize=set_size(width="full", ratio=1.8), 
                 constrained_layout=True)
ax1 = fig.add_subplot(111)
  # instantiate a second axes that shares the same x-axis
a = ax1.bar(POSITIONS, round(100*II_III_corrdata['3 involved']/II_III_corrdata['3 total'],0), color=usz_red, width=WIDTHS, label= "lymph nodes investigated")
ax1.set_xticks(POSITIONS, LABELS)
ax1.set_xlabel("lymph nodes positive level II ipsilateral")
ax1.set_ylabel("involvement level III [%]")

j = 0
for p in a.patches:
    if p.get_height()==0:
      plt.annotate(str(II_III_corrdata['3 involved'][j]) + ' / ' + str(II_III_corrdata['3 total'][j]), 
                  (p.get_x() + p.get_width() / 2., 2), 
                    ha = 'center', va = 'center', 
                    size=7,
                    xytext = (0, 0), 
                    textcoords = 'offset points')
    else:
        plt.annotate(str(II_III_corrdata['3 involved'][j]) + ' / ' + str(II_III_corrdata['3 total'][j]), 
                  (p.get_x() + p.get_width() / 2., p.get_height() / 2.), 
                    ha = 'center', va = 'center', 
                    size=7,
                    xytext = (0, 0), 
                    textcoords = 'offset points')
    if j==4:
        j += 2
    else:
        j += 1

plt.savefig("./figures/II_III_corr_OC.png")

#boxplot for the number of investigated lymph nodes
fig = plt.figure(figsize=set_size(width="full", ratio=1.8), 
                 constrained_layout=True)
ax = fig.add_subplot(111)


b = ax.boxplot(data_raw['total invest'])
ax.set_ylabel("number of lymph nodes")

plt.savefig("./figures/investigated_LN_OC.png")

#generate the plot with the 2d histograms as subplots
# Load the image files
image1 = imread('./figures/lymph_invest_hist2dIb ipsi_OC.png')
image2 = imread('./figures/lymph_invest_hist2dII ipsi_OC.png')
image3 = imread('./figures/lymph_invest_hist2dIII ipsi_OC.png')
image4 = imread('./figures/lymph_invest_hist2dIV ipsi_OC.png')
image5 = imread('./figures/lymph_invest_hist2dV ipsi_OC.png')
image6 = imread('./figures/lymph_invest_hist_OC.png')


# Create a new figure
fig = plt.figure(figsize=set_size(width="full", ratio=1.2),
                 constrained_layout=True, dpi=1000)

# Add subplots and display images
ax1 = fig.add_subplot(3, 2, 1)
ax1.imshow(image1)
ax1.axis('off')

ax2 = fig.add_subplot(3, 2, 2)
ax2.imshow(image2)
ax2.axis('off')

ax3 = fig.add_subplot(3, 2, 3)
ax3.imshow(image3)
ax3.axis('off')

ax4 = fig.add_subplot(3, 2, 4)
ax4.imshow(image4)
ax4.axis('off')

ax5 = fig.add_subplot(3, 2, 5)
ax5.imshow(image5)
ax5.axis('off')

ax6 = fig.add_subplot(3, 2, 6)
ax6.imshow(image6)
ax6.axis('off')

# Show the figure
plt.savefig("./figures/lymph_invest_hist2ds_combined_OC.png")