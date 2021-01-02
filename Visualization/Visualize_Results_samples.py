# This Script plots y_pred vs y_true for all 15s samples

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# working directory
path = "C:\\Python\\Thesis_Work"

# input folder
inp = path + "\\Results\\Processed\\Samples\\"

# output folder
out = path + "\\Plots\\samples _ y_vs_yhat\\"

# read all files alphabetically
for root, dirs, files in os.walk(inp, topdown=True):
    dirs.clear()

# read files
for i in range(len(files)):
    df = pd.read_csv(inp + files[i], sep=";")
    y = np.array(df[['y']].values)
    yhat = np.array(df[['yhat']].values)
    label = np.array(df[['class']])

    # plot perfect result curve
    x = np.linspace(30, 75, 100)

    for j in range(len(label)):
        # if not (label[j] == 'quiet' and y[j] > 35):
        if not (label[j] == 'quiet' and (y[j] > 35 or yhat[j] > 40)):
            plt.scatter(y[j], yhat[j], color='blue', linewidths=1)

    plt.grid(True)
    plt.plot(x, x, color='red')
    plt.title(files[i].split('_')[1])
    plt.xlabel('y_true [dBA]')
    plt.ylabel('y_pred [dBA]')
    plt.savefig(out + files[i].split('_')[1] + "_" + str(int(i/4)+1))    # save with name: model_type + n_iter
    # plt.show()
    plt.close()
