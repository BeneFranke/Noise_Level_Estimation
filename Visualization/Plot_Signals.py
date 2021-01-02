# This script is used for visual presentations of audio data. It is plotting the following types of graphs:
# -> time signal
# -> spectrum
# -> spectrogram
# -> MFFCs

import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display

# working directory
path = "C:\\Python\\Thesis_Work\\"

# audio input folder
input = path + "Audio\\Out\\"

# set input files
file_1 = "quiet _ Benedikt_10 _ 31dBA _ 1580834734203 _ 26 _ 31.1 _ dBA"                # quiet rec with MacBook
file_2 = "quiet _ Franke _ 31dBA _ 1576587475515 _ 41 _ 32.1 _ dBA"                     # quiet rec with Fujitsu
file_3 = "traffic _ 02_Benedikt_F _ 60dBA _ 1581418158044 _ 36 _ 60.6 _ dBA"              # traffic 60dB rec
file_4 = "Radio _ Franke _ 60dBA _ 1576594946057 _ 47 _ 60.1 _ dBA"                     # radio 60dB rec
file_5 = "Music _ Franke _ 60dBA _ 1576665701178 _ 42 _ 60.3 _ dBA"                     # music 60dB rec
file_6 = "TV _ Franke _ 60dBA _ 1576669138941 _ 15 _ 60.3 _ dBA"                     # music 60dB rec


def plot_signal(file1, file2, sr=None):
    # read audio data
    y1, sr1 = librosa.load(file1, sr=sr)
    y2, sr2 = librosa.load(file2, sr=sr)

    # define linspace
    x = np.linspace(0, 15, len(y1))

    # show and save plot
    plt.figure()
    plt.plot(x, y1, color='blue')
    plt.plot(x, y2, color='red')
    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.title('Signal of quiet recording')
    plt.legend(('Hardware A', 'Hardware B'))
    plt.grid(True)
    plt.savefig(path + "Plots\\audio_signal\\time_vs_amp\\" + 'quiet')
    plt.show()


def plot_spectrum(file, sr=None, n_fft=2048, hop_length=None):
    # read audio data
    y, sr = librosa.load(file, sr=sr)

    # convert to frequency dimension
    y = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # define linspace
    x = np.linspace(0, 15, len(y))

    # show and save plot
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("frequency [Hz]")
    plt.xscale("log")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.savefig(path + "Plots\\audio_signal\\frequency_vs_amp\\" + file)
    plt.show()


def plot_spectrogram(file, sr=None,  n_fft=2048, hop_length=None, n_mels=64):
    # read audio data
    y, sr = librosa.load(file, sr=sr)
    # extract mel amplitudes
    specs = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    # convert to dB
    specs_db = librosa.power_to_db(specs)

    # Plot spectrum
    librosa.display.specshow(specs_db, sr=sr, y_axis='mel', x_axis='time')
    plt.title("Mel-Spectrogram of " + os.path.basename(file).split(' _ ')[0] + " sample")

    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path + "Plots\\audio_signal\\spectrogram\\mel_spec_" + os.path.basename(file).split(' _ ')[0])
    plt.show()


def plot_mfcc(file, scenario, sr=None, n_mffc=23):
    # read audio data
    y, sr = librosa.load(file, sr=sr)

    # extract mfccs and normalize
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mffc) #, norm='ortho')
    # mfcc = librosa.util.normalize(mfcc)

    # define linspace
    x = np.linspace(0, 15, len(mfcc.T))

    # show and save plot
    plt.figure()
    plt.plot(x, mfcc.T)
    # plt.ylim((-1.1, 1.1))
    plt.ylim((-600, 600))
    plt.xlabel("Time [s]")
    plt.ylabel("MFCCs")
    # plt.title(str(n_mffc) + " MFCCs of " + scenario + " sample")
    plt.title("MFCCs of " + scenario + " sample")
    plt.grid(True)
    plt.savefig(path + "Plots\\audio_signal\\mfcc\\mfcc_" + os.path.basename(file).split(' _ ')[0])
    plt.show()


if __name__ == "__main__":
    # plot time signal
    plot_signal(input + file_1 + ".wav", input + file_2 + ".wav")

    # plot frequency signal
    # plot_spectrum(input + "")
    # plot_spectrum(input + "")

    # plot spectrogram
    """plot_spectrogram(input + file_1 + ".wav")
    plot_spectrogram(input + file_3 + ".wav")
    plot_spectrogram(input + file_5 + ".wav")
    plot_spectrogram(input + file_6 + ".wav")"""

    """"# plot MFCCs
    plot_mfcc(input + file_1 + ".wav", scenario="quiet")
    plot_mfcc(input + file_3 + ".wav", scenario="traffic")
    plot_mfcc(input + file_5 + ".wav", scenario="music")
    plot_mfcc(input + file_6 + ".wav", scenario="TV")"""
