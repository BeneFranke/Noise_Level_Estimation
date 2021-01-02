# Extract MFCC features from Audio Signal

import librosa
import os
import numpy as np
import pandas as pd
import timeit
import multiprocessing as mp
from os.path import basename, splitext

# working directory
path = "C:\\Python\\Thesis_Work"

# audio directory
audio_path = "C:\\Python\\Thesis_Work\\Audio\\Out\\"

# read all files alphabetically
for root, dirs, files in os.walk(audio_path, topdown=True):
    dirs.clear()

# Set number of coefficients
N_MFCC = 23


def extract_mfcc(file):
    # read audio
    y, sr = librosa.load(audio_path + file, duration=15, sr=None)

    # extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, S=None, n_mfcc=N_MFCC, dct_type=2)

    # normalize
    mfcc = librosa.util.normalize(mfcc)

    # swap axes -> row = time frames , columns = features
    mfcc = np.transpose(mfcc)

    # read spl, label and file name
    spl = np.full((np.shape(mfcc)[0], 1), file.split(' _ ')[5])
    label = np.full((np.shape(mfcc)[0], 1), file.split(' _ ')[0])
    name = np.full((np.shape(mfcc)[0], 1), splitext(basename(file))[0])

    return np.c_[mfcc, spl, label, name]                               # shape: (1407, 26)


if __name__ == "__main__":
    # start timer
    start = timeit.default_timer()

    # compute multiprocessing on mfcc extraction
    with mp.Pool(mp.cpu_count()) as p:
        data = p.map(extract_mfcc, files)                             # shape: (len(files), 1407, 16)
        p.close()

    # print mfcc extraction time
    stop1 = timeit.default_timer()
    print('MFCC Extraction Time: ', round(stop1 - start, 3), 's')

    # write column headings
    features = []
    for i in range(N_MFCC):
        features.append("MFCC_Feature_" + str(i+1))
    info = ["SPL_dBA", "Label", "File_Name"]

    # reshape output dimensions to (len(files) * number_frames, features + info)
    data = np.asarray(data).reshape((len(files)*1407, N_MFCC+3))

    # create DataFrame and save as .csv
    df = pd.DataFrame(data, columns=features+info)
    df.to_csv(path+"\\Features\\MFCC_" + str(N_MFCC) + ".csv", index=False, sep=";")

    # print write dataframe time
    stop2 = timeit.default_timer()
    print('Write DataFrame Time: ', round(stop2 - stop1 - start, 3), 's')
