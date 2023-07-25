'''
This code creates Medium mass and 
copy the Medium mass to the desired folder
'''
import os
import shutil
import pandas as pd


data = pd.read_csv('C:/Users/zahra/Desktop/FarzanehFiles/Codes/data.csv', usecols=['MRN', 'Diagnosis', 'Type_of_Mass'])
positive_folder = 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/2_DataLabeling/CleanedData/Positive'
output_folder = 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/2_DataLabeling/CleanedData/medium_mass/Positive'

medium_mass = data[(data['Diagnosis'] == 'Mass') & (data['Type_of_Mass'].isin(['Nodule', 'Cyst', 'Lesion', 'Amyloid']))].copy()
medium_mass.dropna(inplace=True)
medium_mass['label'] = 'Mass'
medium_mass_count = len(medium_mass)

print(medium_mass)
print("Number of medium_mass in csv file:", medium_mass_count)

#%% copy the medium mass to the desired folder

counter = 0
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
mrns = medium_mass['MRN'].tolist()

file_dict = {}
for filename in os.listdir(positive_folder):
    file_path = os.path.join(positive_folder, filename)
    if os.path.isfile(file_path):
        mrn = filename.split(".")[0]
        if mrn in mrns:
            if mrn not in file_dict:
                file_dict[mrn] = 1
            else:
                file_dict[mrn] += 1
            new_filename = f"{mrn}_{file_dict[mrn]}{os.path.splitext(filename)[1]}"
            output_path = os.path.join(output_folder, new_filename)
            shutil.copyfile(file_path, output_path)
            print(f"Copied file: {filename} as {new_filename}")
            counter += 1
print(f"Total samples copied: {counter}")