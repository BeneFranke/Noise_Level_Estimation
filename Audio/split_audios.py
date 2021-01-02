# This script splits up audio files into 15 sec samples

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
import timeit

# input folder
audio_path = 'C:\\Python\\Thesis_Work\\Audio\\In\\'
levels_path = 'C:\\Python\\Thesis_Work\\Audio\\SPL_meter\\'

# output folder
splitted_wav = 'C:\\Python\\Thesis_Work\\Audio\\Out\\'

# time of single audio sample [s]
frame_length = 15


def read_audios(file):
    print(splitext(basename(file))[0])

    # read spl values
    for root, dirs, files in os.walk(levels_path):
        for name in files:
            name

    spl = np.zeros(64)

    # read spl-data and write to file name
    for i in range(len(files)):
        if splitext(basename(file))[0] == splitext(files[i])[0]:
            spl = np.loadtxt(levels_path+files[i], delimiter=";")

    y, sr = librosa.load(file, sr=None, mono=False)

    y_mono = librosa.to_mono(y)

    # Save as .wav
    for i in range(int(900 * sr // (sr * frame_length))):       # for first 900 sec = 15 min
        yx = y_mono[int(i*sr*frame_length):int(sr*frame_length*(i+1))]
        soundfile.write(splitted_wav + file.split('\\')[-1].split('.webm')[0] + ' _ ' + str(i+1) + ' _ ' + str(spl[i])
                        + ' _ ' + 'dBA' + '.wav', yx, sr, 'PCM_16')


if __name__ == '__main__':

    P = Pool(mp.cpu_count())
    P.map(read_audios, glob(audio_path + '/*.webm'))
    P.close()
