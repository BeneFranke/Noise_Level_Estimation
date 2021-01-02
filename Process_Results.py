# This script takes testing results of models and average RMSE over
# - each 15sec sample
# - each 15min recording
# - each class
# - all data

import numpy as np
import pandas as pd
import os

# working directory
path = "C:\\Python\\Thesis_Work\\Results\\"

# read all files alphabetically
for root, dirs, files in os.walk(path + "n-times-results\\", topdown=True):
    dirs.clear()


# define function for difference calculation
def dif(val1, val2):
    return np.sqrt((val1-val2)**2)


for j in range(len(files)):
    # load data
    df = pd.read_csv(path + "n-times-results\\" + files[j], sep=";")
    y = np.array(df[['y_true']].values)
    yhat = np.array(df[['y_pred']].values)
    y_dif = np.array(df[['y_dif']].values)
    label = np.array(df[['class']])
    sample = np.array(df[['recording']])

    # create empty arrays and file counter
    y_sample = np.empty((0, 6), dtype=object)  # [y, yhat, y_dif_sample, y_dif_frame, class, sample]
    y_rec = np.empty((0, 5), dtype=object)  # [y, yhat, y_dif_sample, class, sample]
    y_class = np.empty((0, 2), dtype=object)  # [y_dif_sample, class]

    # first loop: average over samples
    ysum = yhat_sum = ydif_sum = count = 0
    print("STEP ONE: SAMPLES")
    for i in range(len(y)):
        ysum += y[i]
        yhat_sum += yhat[i]
        ydif_sum += y_dif[i]
        count += 1

        if i == len(y)-1:
            y_sample = np.vstack((y_sample, np.r_[ysum/count, yhat_sum/count, dif(ysum/count, yhat_sum/count),
                                                  ydif_sum/count, label[i], sample[i]]))
            # print(count)
            ysum = yhat_sum = ydif_sum = count = 0

        elif sample[i] != sample[i+1]:
            y_sample = np.vstack((y_sample, np.r_[ysum/count, yhat_sum/count, dif(ysum/count, yhat_sum/count),
                                                  ydif_sum/count, label[i], sample[i]]))
            # print(count)
            ysum = yhat_sum = ydif_sum = count = 0

    # second loop: average over recordings
    ysum = yhat_sum = ydif_sum = count = 0
    print("STEP TWO: RECORDINGS")
    for i in range(len(y_sample)):
        ysum += y_sample[i][0]
        yhat_sum += y_sample[i][1]
        ydif_sum += y_sample[i][2]
        count += 1

        if i == len(y_sample)-1:
            y_rec = np.vstack((y_rec, [ysum/count, yhat_sum/count, ydif_sum/count, y_sample[i][4],
                                       ' _ '.join(str(y_sample[i][5]).split(' _ ')[0:4])]))
            # print(count)
            ysum = yhat_sum = ydif_sum = count = 0

        elif ' _ '.join(str(y_sample[i][5]).split(' _ ')[0:4]) != ' _ '.join(str(y_sample[i + 1][5]).split(' _ ')[0:4]):
            y_rec = np.vstack((y_rec, [ysum/count, yhat_sum/count, ydif_sum/count, y_sample[i][4],
                                       ' _ '.join(y_sample[i][5].split(' _ ')[0:4])]))
            # print(count)
            ysum = yhat_sum = ydif_sum = count = 0

    # third loop: average over classes
    ysum = yhat_sum = ydif_sum = count = 0
    print("STEP THREE: CLASSES")
    for i in range(len(y_rec)):
        ydif_sum += float(y_rec[i][2])
        count += 1

        if i == len(y_rec)-1:
            y_class = np.vstack((y_class, [ydif_sum/count,  y_rec[i][3]]))
            ydif_sum = count = 0

        elif y_rec[i][3] != y_rec[i+1][3]:
            y_class = np.vstack((y_class, [ydif_sum/count,  y_rec[i][3]]))
            ydif_sum = count = 0

    # save results
    df_sample = pd.DataFrame(data=y_sample, columns=["y", "yhat", "y_dif_sample", "y_dif_frame", "class",
                                                     "recording sample"])
    df_rec = pd.DataFrame(data=y_rec, columns=["y", "yhat", "y_dif_sample",  "class", "recording"])
    df_class = pd.DataFrame(data=y_class, columns=["y_dif_sample", "class"])

    df_sample.to_csv(path+"\\Processed\\" + files[j] + "_processed_samples.csv", index=False, sep=";")
    df_rec.to_csv(path+"\\Processed\\" + files[j] + "_processed_recordings.csv", index=False, sep=";")
    df_class.to_csv(path+"\\Processed\\" + files[j] + "_processed_classes.csv", index=False, sep=";")

    print(j+1, "out of ", len(files), "files ---", 100*(j+1)/len(files), "% --- Done")
