# create Mel-Spectrograms from Audio Signal

import numpy as np
import pandas as pd
import os
from os.path import basename, splitext
import multiprocessing as mp
import librosa
import librosa.display
import timeit

# working directory
path = "C:\\Python\\Thesis_Work"

# audio directory
audio_path = "C:\\Python\\Thesis_Work\\Audio\\Out\\"

# read all files alphabetically
for root, dirs, files in os.walk(audio_path, topdown=True):
    dirs.clear()

# parameters
N_MELS = 64
n_fft = 1024
hop_length = 512
Sr = None


def extract_mel_spec(file):
    # read audio
    y, sr = librosa.load(audio_path + file, Sr)      # sr=None -> 48000 1/s , 22050 1/s -> 646 frames

    # extract mel amplitudes
    specs = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, win_length=n_fft,
                                           hop_length=hop_length, n_mels=N_MELS) #, fmax=8000)

    # convert to dB
    specs_db = librosa.power_to_db(specs)

    # write to file
    df = pd.DataFrame(specs_db)
    df.to_csv(path + "\\Features\\Mel_Spectrogram_dB_" + str(N_MELS) + "_1407\\"
              + splitext(basename(file))[0] + ".csv", index=False, sep=";")

    return specs_db                             # shape: (128, 1407)


if __name__ == "__main__":
    # start timer
    start = timeit.default_timer()

    # compute multiprocessing on mfcc extraction
    with mp.Pool(mp.cpu_count()) as p:
        p.map(extract_mel_spec, files)
        p.close()

    # print mel-features extraction time
    stop = timeit.default_timer()
    print('Mel-features Extraction Time: ', round(stop - start, 3), 's')
