# This script extracts the spl values from the spl-meter-txt-files and saves them as csv for further processing
import csv
import numpy as np
import os

# input folder
fol_in = "C:\\Python\\Thesis_Work\\Audio\\SPL_meter\\In\\"

# output folder
fol_out = "C:\\Python\\Thesis_Work\\Audio\\SPL_meter\\"

# read all file names
for root, dirs, files in os.walk(fol_in):
    for name in files:
        name

print('number of files: ', len(files))

for file in files[:len(files)]:
    f = open(fol_in+file, 'rt')
    spl = []

    num_rows = 0
    for item in csv.reader(f):                              # read any row
        if f.read(20)[15:20] == '':
            break
        spl.append(float(f.read(20)[15:20]))                # read correct column and save value as float
        num_rows += 1

    print(file, " - processed - ", "number rows:", num_rows)

    SPL = np.asarray(spl)

    SPL_out = np.zeros(64)

    if num_rows == 192:   # for 5s measurement intervals average spl to 15s intervals
        j = 0
        for i in range(0, num_rows, 3):
            SPL[i] = (SPL[i]+SPL[i+1]+SPL[i+2])/3
            if i > 0:
                SPL[i-1] = 0
                SPL[i-2] = 0
            if i == num_rows-3:
                SPL[i+1] = 0
                SPL[i+2] = 0

        for i in range(0, num_rows, 1):
            if SPL[i] != 0:
                SPL_out[j] = round(SPL[i], 1)
                j += 1

    elif num_rows == 64:
        SPL_out = SPL

    else:
        print("ERROR: number of rows not compatible")

    # save as csv
    np.savetxt(fol_out+file, SPL_out, delimiter=";", fmt='%2.5f')
