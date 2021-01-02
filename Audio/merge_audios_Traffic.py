# This script merges two 20 or 30 sec samples randomly ordered into one 3 minute audio file

import numpy as np
import pandas as pd
from glob import glob
import librosa
import soundfile
from multiprocessing import Pool
import multiprocessing as mp
import os
from os.path import basename, splitext
import csv
import random

# input folder
audio_path = 'C:\\Users\\mail\\OneDrive\\Desktop\\Audio\\Traffic_Noise_Synthesis\\In\\'

# output folder
merge_wav = 'C:\\Users\\mail\\OneDrive\\Desktop\\Audio\\Traffic_Noise_Synthesis\\'

y_a, sr_a = librosa.load(audio_path + "A_org.wav", sr=None, mono=False)
y_b, sr_b = librosa.load(audio_path + "B_org.wav", sr=None, mono=False)
y_a = y_a[:, int(sr_a*2):int(sr_a*21)]    # cut signal A to 20s file

for k in range(1):                        # create X * 3 min samples

    out_str = ""
    time = 0
    Array = ['A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B']           # for 3 minutes: 9 * 20s frames
    random.shuffle(Array)

    for i in range(9):
        if Array[i] == 'A':
            if i == 0:
                audio_data = y_a
            else:
                yNew = y_a
            out_str = out_str + "A"
            time = time + 20 #+30
        else:
            if i == 0:
                audio_data = y_b
            else:
                yNew = y_b
            out_str = out_str + "B"
            time = time + 20

        if i > 0:
            audio_data = np.hstack((audio_data, yNew))          # add data to stack

        if time >= 180:                     # break when 3 min play time is reached
            break

    librosa.output.write_wav(merge_wav + str(k) + " " + out_str + ".wav", audio_data, sr_b)         # write wav-file

"""    while True:                             # further additions

        flip = random.randint(0, 1)         # randomly choose sample A or B
        if flip == 1:
            yNew = y_a
            out_str = out_str + "A"
            time = time + 30
        else:
            yNew = y_b
            out_str = out_str + "B"
            time = time + 20
"""