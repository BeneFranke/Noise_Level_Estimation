# This script read labels from file name and saves labels and categories as csv-file
import numpy as np
import os

# input folder
fol_in = "C:\\Python\\Thesis_Work\\Audio\\Out\\"

# output folder
fol_out = "C:\\Python\\Thesis_Work\\Features\\"

# read all file names
for root, dirs, files in os.walk(fol_in):
    for name in files:
        name

# initialize variables
category = ["" for x in range(len(files))]      # Category equals: Melodic, Mechanic, Quiet
label = ["" for x in range(len(files))]         # Label is more specific: TV, Music, Radio, Autos, CoffeeMachine..
i = mech = mel = quiet = 0

for file in files[:len(files)]:
    # read labels
    label[i] = file.split(' ')[0]

    # set categories
    if label[i] == 'quiet':
        category[i] = 'Quiet'
        quiet += 1

    elif label[i] == 'TV' or label[i] == 'Radio' or label[i] == 'Music':
        category[i] = 'Harmonic'
        mel += 1

    else:
        category[i] = 'Mechanic'
        mech += 1

    i += 1

print('------------------------------')
print('number of files: ', len(files))
print('------------------------------')
print("Labels: ", label)
print("Categories: ", category)
print('------------------------------')
print("number Mechanic sounds:", mech)
print("number Melodic sounds:", mel)
print("number Quiet sounds:", quiet)
print('------------------------------')

# convert to MFCC time frames - 646 frames each 15s
Cat_MFCC = np.empty([(len(files) * 646), 2], dtype=str)               # create empty matrix
for i in range(len(files)):                                           # set labels for 646 frames per file
    k = 646 * i
    for j in range(646):
        Cat_MFCC[j + k, 0] = str(category[i])                         # save Categories in 1st row
        Cat_MFCC[j + k, 1] = str(label[i])                            # save Labels in 2nd row

# save as csv
np.savetxt(fol_out+"Categories_MFCC.csv", Cat_MFCC, delimiter=";", fmt='%s')
np.savetxt(fol_out+"Labels.csv", label, delimiter=";", fmt='%s')
np.savetxt(fol_out+"Categories.csv", category, delimiter=";", fmt='%s')

# Just saves first character instead of whole string -> FIX IT!!
