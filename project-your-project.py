#!/usr/bin/env python
# coding: utf-8

# # project-your-project

#   `project-your-project` is a simple tool to visualize your project's
#   progress. `project-your-project` generates a Gannt chart that displays both
#   the baseline and current schedule to easily visualize if your project is
#   projected to be on track or not.
#
#   This project uses the excellent tutorial as the base and adds customizations:
# 
#   https://towardsdatascience.com/gantt-charts-with-pythons-matplotlib-395b7af72d72
# 

# ## Setup

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.patches import Patch
from datetime import datetime

# ### Parse and read arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputfile", type=str, help="XLSX inputfile", required=True)
parser.add_argument("-t", "--title", type=str, help="Project's title", required=True)
parser.add_argument("-o", "--outfilepre", type=str, help="Project's timeline prefix in JPG format", required=True)
parser.add_argument("-c", "--categories", nargs='+', help="Project's categories", required=True)
parser.add_argument("-x", "--hexcolors", nargs='+', help="Project's colors per category", required=True)
parser.add_argument("-b", "--barheights", nargs='+', type=float, help="Project's bar heights per category", required=True)
args = parser.parse_args()

SMALL_SIZE = 10
MEDIUM_SIZE = 11
BIGGER_SIZE = 16
dateformat = '%Y%m%d_%H%M' # e.g., 20220519_2334

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# set up project specific values
proj_infile = args.inputfile
proj_title = args.title
proj_outfile = args.outfilepre + datetime.strftime(datetime.now(),dateformat) + '.jpg'
proj_categories = args.categories
proj_colors = args.hexcolors
proj_heights = args.barheights

# ## Data Processing

# Read input XLSX file 
df = pd.read_excel(proj_infile)
print(df)

# project start date
proj_start = df.Start.min()

# number of days from project start to task start
df['start_num'] = (df.Start-proj_start).dt.days

# number of days from project start to end of tasks
df['end_num'] = (df.End-proj_start).dt.days

# days between start and end of each task
df['days_start_to_end'] = df.end_num - df.start_num

# days between start and current progression of each task
df['current_num'] = (df.days_start_to_end * df.Completion)


# # Create Figure

# create a column with the color for each department
def color(row):
    c_dict = dict(zip(proj_categories, proj_colors))
    return c_dict[row['Category']]
df['color'] = df.apply(color, axis=1)

# create a column with the height for each bar
def height(row):
    h_dict = dict(zip(proj_categories, proj_heights))
    return h_dict[row['Category']]
df['height'] = df.apply(height, axis=1)


##### PLOT #####
fig, (ax, ax1) = plt.subplots(2, figsize=(28,16), gridspec_kw={'height_ratios':[6, 1]})

# bars
ax.barh(df.Task, df.current_num, left=df.start_num, color=df.color, height=df.height)
ax.barh(df.Task, df.days_start_to_end, left=df.start_num, color=df.color, alpha=0.5, height=df.height)
ax.invert_yaxis()

for idx, row in df.iterrows():
    ax.text(row.end_num+0.1, idx, f"{int(row.Completion*100)}%", va='center', alpha=0.8)
    ax.text(row.start_num-0.1, idx, row.Task, va='center', ha='right', alpha=0.8)


# grid lines
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.2, which='both')

# ticks
xticks = np.arange(0, df.end_num.max()+1, 3)
xticks_labels = pd.date_range(proj_start, end=df.End.max()).strftime("%m/%d")
xticks_minor = np.arange(0, df.end_num.max()+1, 1)
ax.set_xticks(xticks)
ax.set_xticks(xticks_minor, minor=True)
ax.set_xticklabels(xticks_labels[::3])
ax.set_yticks([])

# ticks top
# create a new axis with the same y
ax_top = ax.twiny()

# align x axis
ax.set_xlim(0, df.end_num.max())
ax_top.set_xlim(0, df.end_num.max())

# top ticks (markings)
xticks_top_minor = np.arange(0, df.end_num.max()+1, 7)
ax_top.set_xticks(xticks_top_minor, minor=True)
# top ticks (label)
xticks_top_major = np.arange(3.5, df.end_num.max()+1, 7)
ax_top.set_xticks(xticks_top_major, minor=False)
# week labels
xticks_top_labels = [f"Week {i}"for i in np.arange(1, len(xticks_top_major)+1, 1)]
ax_top.set_xticklabels(xticks_top_labels, ha='center', minor=False)

# hide major tick (we only want the label)
ax_top.tick_params(which='major', color='w')
# increase minor ticks (to marks the weeks start and end)
ax_top.tick_params(which='minor', length=8, color='k')

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['left'].set_position(('outward', 10))
ax.spines['top'].set_visible(False)

ax_top.spines['right'].set_visible(False)
ax_top.spines['left'].set_visible(False)
ax_top.spines['top'].set_visible(False)

plt.suptitle(proj_title)

##### LEGENDS #####
c_dict = dict(zip(proj_categories, proj_colors))
legend_elements = [Patch(facecolor=c_dict[i], label=i)  for i in c_dict]

ax1.legend(handles=legend_elements, loc='upper center', ncol=4, frameon=False)

# clean second axis
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])

#plt.show()


fig.savefig(proj_outfile) 

