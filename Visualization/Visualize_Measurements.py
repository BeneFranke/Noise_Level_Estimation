# This Script plots the measured sound level over the recording duration

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time

# working directory
path = "/Audio/SPL_meter"

# read all files
for root, dirs, files in os.walk(path):
    for name in files:
        name

# calc mean and dif of set spl
for i in range(len(files)):
    # read spl from file
    spl = np.loadtxt(path + '\\' + files[i], delimiter=";")

    # calculate mean and difference to set spl
    spl_mean = round(np.mean(spl), 2)
    spl_set = int(files[i].split(' _ ')[2][0:2])
    spl_dif = round(np.sqrt((spl_set - spl_mean) ** 2), 2)
    print(files[i].split(' _ ')[0], ' ', files[i].split(' _ ')[2],  "---", "SPL set:",
          spl_set, " SPL real:", spl_mean, " Difference:", spl_dif)
"""
# plot spl of each recording separately
for i in range(len(files)):
    # read spl for 3 files
    spl= np.loadtxt(path + '\\' + files[i], delimiter=";")

    x_scale = np.arange(0, 16, 0.25)

    # plot spl over recording time
    plt.plot(x_scale, spl, color='blue')
    if files[i].split(' _ ')[0] != 'quiet':             # don`t plot legend for quiet scenario
        plt.legend(('60 dBA', '60 dBA', '70 dBA'))
    plt.title(files[i].split(' _ ')[0])
    plt.xlabel("Recording time [min]")
    plt.ylabel('SPL [dBA]')
    plt.grid(True)
    plt.ylim((20, 100))
    plt.xlim(0, 15.25)                          # wertebereich 0 bis 63 = 16 min , 0 bis 59 = 15 min
    plt.show()
"""
# plot spl of each class in one plot
spl = np.zeros((64, 3))
for i in range(0, len(files), 3):
    # read spl for 3 files
    for j in range(3):
        spl[:, j] = np.loadtxt(path + '\\' + files[i+j], delimiter=";")

    x_scale = np.arange(0, 16, 0.25)

    # plot spl over recording time
    plt.plot(x_scale, spl[:, 0], color='green')
    plt.plot(x_scale, spl[:, 1], color='blue')
    plt.plot(x_scale, spl[:, 2], color='red')
    if files[i].split(' _ ')[0] != 'quiet':             # don`t plot legend for quiet scenario
        plt.legend(('50 dBA', '60 dBA', '70 dBA'))
    plt.title(files[i].split(' _ ')[0])
    plt.xlabel("Recording time [min]")
    plt.ylabel('SPL [dBA]')
    plt.grid(True)
    plt.ylim((20, 100))
    plt.xlim(0, 15.25)                          # wertebereich 0 bis 63 = 16 min , 0 bis 59 = 15 min
    plt.show()



