'''
This code creates big mass and 
copy the big mass to the desired folder
'''

import os
import shutil
import pandas as pd

data = pd.read_csv('C:/Users/zahra/Desktop/FarzanehFiles/Codes/data.csv', usecols=['MRN', 'Diagnosis', 'Type_of_Mass'])
positive_folder = 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/2_DataLabeling/CleanedData/Positive'
output_folder = 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/2_DataLabeling/CleanedData/big_mass/Positive'

big_mass = data[(data['Diagnosis'] == 'Mass') & (data['Type_of_Mass'].isin(['Papilloma', 'Cancer', 'Polyp']))].copy()
big_mass.dropna(inplace=True)
big_mass['label'] = 'Mass'
big_mass_count = len(big_mass)
print(big_mass)
print("Number of big_mass in csv file:", big_mass_count)

#%% Copy the Big mass to the desired folder
counter = 0
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
mrns = big_mass['MRN'].tolist()

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

#%% 
'''
def count_items_in_folder(folder_path):
    count = 0
    for _, _, files in os.walk(folder_path):
        count += len(files)
    return count

folder_path = "C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/2_DataLabeling/CleanedData/Positive" 
item_count = count_items_in_folder(folder_path)
print("Number of items in the folder:", item_count)
'''