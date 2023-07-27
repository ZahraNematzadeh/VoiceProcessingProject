'''
This code creates big mass and 
copy the big mass to the desired folder
'''

import os
import shutil
import pandas as pd

data = pd.read_csv('C:/Users/zahra/VoiceColab - Copy/dataset/metadata/data.csv', usecols=['MRN', 'Diagnosis', 'Type_of_Mass'])
positive_folder = 'C:/Users/zahra/Desktop/FarzanehFiles/DataFolders/Wav_format/SplittedData/1_e/Positive'
output_folder = 'C:/Users/zahra/VoiceColab - Copy/dataset/e/test_train/ClusteredData/big_mass_wav/Positive'

if not os.path.exists(output_folder):
        os.makedirs(output_folder)

big_mass = data[(data['Diagnosis'] == 'Mass') & (data['Type_of_Mass'].isin(['Papilloma', 'Cancer', 'Polyp']))].copy()
big_mass.dropna(inplace=True)
big_mass['label'] = 'Mass'
big_mass_count = len(big_mass)
print(big_mass)
print("Number of big_mass in csv file:", big_mass_count)
#------------------------------------------------------
counter = 0
mrns = big_mass['MRN'].tolist()
file_dict = {}
for filename in os.listdir(positive_folder):
    file_path = os.path.join(positive_folder, filename)
    if os.path.isfile(file_path):
        mrn = filename.split(".")[0]
        first_dash_index = mrn.find('-')
        desired_substring = mrn[first_dash_index + 1:]
        if desired_substring in mrns:
            output_path = os.path.join(output_folder, filename)
            shutil.copyfile(file_path, output_path)
            counter += 1
print(f"Total samples copied: {counter}")

