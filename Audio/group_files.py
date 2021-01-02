# This script read labels from file name and saves labels and categories as csv-file
import numpy as np
import os
from os.path import basename, splitext

# input folder
fol_in = "C:\\Python\\Thesis_Work\\Audio\\In\\"

# output folder
fol_out = "C:\\Python\\Thesis_Work\\Features\\"

# read all file names
for root, dirs, files in os.walk(fol_in):
    for name in files:
        name

# initialize variables
group = ["" for x in range(len(files))]
num_frames = 646 * 60                                                      # number frames of each recording (15min)
groups_mfcc = ["" for x in range(len(files) * num_frames)]
i = 0

# read file name of each recording and extract individual code
for file in files[:len(files)]:
    group[i] = splitext(basename(file))[0]
    i += 1

# save group for each recording and match with MFCC frame size
for i in range(len(files)):
    k = i * num_frames
    for j in range(num_frames):
        groups_mfcc[j + k] = group[i]

# save as csv
np.savetxt(fol_out+"Groups.csv", group, delimiter=";", fmt='%s')
np.savetxt(fol_out+"Groups_MFCC.csv", groups_mfcc, delimiter=";", fmt='%s')
