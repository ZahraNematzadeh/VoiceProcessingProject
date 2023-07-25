'''
This code creates small mass and 
copy the small mass to the desired folder
'''
import os
import shutil
import pandas as pd

data = pd.read_csv('C:/Users/zahra/Desktop/FarzanehFiles/Codes/data.csv', usecols=['MRN', 'Diagnosis', 'Type_of_Mass'])
positive_folder = 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/2_DataLabeling/CleanedData/Positive'
output_folder = 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/2_DataLabeling/CleanedData/small_mass/Positive'

small_mass = data[(data['Diagnosis'] == 'Mass') & (data['Type_of_Mass'].isin(['Leukoplakia', 'Dysplasia']))].copy()
small_mass.dropna(inplace=True)
small_mass['label'] = 'Mass'
small_mass_count = len(small_mass)

print(small_mass)
print("Number of small_mass in csv file:", small_mass_count)
#%% copy the small mass to the desired folder
counter = 0
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
mrns = small_mass['MRN'].tolist()

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