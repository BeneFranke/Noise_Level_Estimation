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
audio_path = 'C:\\Users\\mail\\OneDrive\\Desktop\\Audio\\Mechanic_Noise_Synthesis\\In\\'

# output folder
merge_wav = 'C:\\Users\\mail\\OneDrive\\Desktop\\Audio\\Mechanic_Noise_Synthesis\\'

y_a, sr_a = librosa.load(audio_path + "E_WashingMachine.wav", sr=None, mono=False)
y_dish, sr_b = librosa.load(audio_path + "B_Dishwasher.wav", sr=None, mono=False)
#y_c, sr_c = librosa.load(audio_path + "G_Dishwasher.wav", sr=None, mono=False)
#y_a = librosa.to_mono(y_a)
#y_b = librosa.to_mono(y_b)
#y_a = y_a[int(sr_a*122):int(sr_a*142)]     # cut file A
y_b = y_dish[int(sr_b*50):int(sr_b*65)]       # cut file B
y_c = y_dish[int(sr_b*90):int(sr_b*105)]

for k in range(3):                        # create X * 3 min samples

    out_str = ""
    time = 0
    # Array = ['A', 'A', 'A', 'B', 'B', 'B']                        # for 30s clips; 6 * 30s frames
    Array = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'A', 'C']           # for 3 minutes: 9 * 20s frames
    random.shuffle(Array)

    for i in range(12):
        if Array[i] == 'A':
            if i == 0:
                audio_data = y_a
            else:
                yNew = y_a
            out_str = out_str + "A"
            time = time + 20

        elif Array[i] == 'B':
            if i == 0:
                audio_data = y_b
            else:
                yNew = y_b
            out_str = out_str + "B"
            time = time + 15

        elif Array[i] == 'C':
            if i == 0:
                audio_data = y_c
            else:
                yNew = y_c
            out_str = out_str + "C"
            time = time + 15

        if i > 0:
            audio_data = np.hstack((audio_data, yNew))          # add data to stack

        #if time >= 180:                     # break when 3 min play time is reached
        #    break

    # write wav-file
    librosa.output.write_wav(merge_wav + "Mechanic_" + str(k) + "_" + out_str + ".wav", audio_data, sr_b)


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
""" # old randomilization