# This Script plots y_pred vs y_true for all feature frames

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# working directory
path = "C:\\Python\\Thesis_Work"

# input folder
inp = path + "\\Results\\n-times-results\\doorslap_cut\\"

# output folder
out = path + "\\Plots\\frames _ y_vs_yhat\\"

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

    # plot y_true vs y_pred
    for j in range(len(label)):
        plt.scatter(y[j], yhat[j], color='blue', linewidths=1)

    plt.grid(True)
    plt.plot(x, x, color='red')
    plt.title(files[i].split('_')[1])
    plt.xlabel('y_true [dBA]')
    plt.ylabel('y_pred [dBA]')
    plt.savefig(out + files[i].split('_')[1] + "_" + str(int(i/4)+1))
    # plt.show()
    plt.close()

    print(files[i], "DONE", "---", i+1, "out of", len(files), "(", round(100 * (i+1)/len(files), 2), "%) Done")

