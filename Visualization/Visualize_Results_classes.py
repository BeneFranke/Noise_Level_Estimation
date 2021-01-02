# This Script plots y_pred vs y_true for all 15s samples with labelled classes

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# working directory
path = "C:\\Python\\Thesis_Work"

# input folder
inp = path + "\\Results\\Processed\\Samples\\"

# output folder
out = path + "\\Plots\\samples _ y_vs_yhat\\classes\\"

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

    # scatter data points -> classes in different colors
    for j in range(len(label)):
        if label[j] == 'TV':
            s1 = plt.scatter(y[j], yhat[j], c='orange', linewidths=1)
        elif label[j] == 'Music' or label[j] == 'Radio':
            s2 = plt.scatter(y[j], yhat[j], c='green', linewidths=1)
        elif label[j] == 'quiet':
            if not (y[j] > 35 or yhat[j] > 40):             # !!! CUT OUT FOR PRESENTATION !!!
                s3 = plt.scatter(y[j], yhat[j], c='purple', linewidths=1)
        else:
            s4 = plt.scatter(y[j], yhat[j], c='blue', linewidths=1)

    plt.legend((s1, s2, s3, s4), ('TV', 'Music & Radio', 'Quiet', 'Mechanic Noise'))
    plt.grid(True)
    plt.plot(x, x, color='red')
    plt.title(files[i].split('_')[1])
    plt.xlabel('y_true [dBA]')
    plt.ylabel('y_pred [dBA]')
    plt.savefig(out + files[i].split('_')[1] + "_" + str(int(i/4)+1))    # save with name: model_type + n_iter
    # plt.show()
    plt.close()
