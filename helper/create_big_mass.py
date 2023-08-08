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
#---------------------------------------------------------------
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
#================================================================
#================= Preparing and copying Negative folder=========
negative_folder = 'C:/Users/zahra/Desktop/FarzanehFiles/DataFolders/Wav_format/SplittedData/1_e/Negative'
neg_output_folder = 'C:/Users/zahra/VoiceColab - Copy/dataset/e/test_train/ClusteredData/big_mass_wav/Negative'
count_reinke = 0

if not os.path.exists(neg_output_folder):
        os.makedirs(neg_output_folder)
        
for filename in os.listdir(negative_folder):
    file_path = os.path.join(negative_folder, filename)
    if os.path.isfile(file_path):
        output_path = os.path.join(neg_output_folder, filename)
        shutil.copyfile(file_path, output_path)
        print(f"Copied file: {filename}")
#====================================================================        
#finding Reinke Edema in Diagnosis and adding it to negative folder        
new_other = data[(data['Diagnosis'] == 'Mass') & (data['Type_of_Mass'].isin(["Reinke's Edema"]))].copy()        
new_other_set = set(new_other['MRN'])
count_reinke = 0

for filename in os.listdir(positive_folder):
    file_path = os.path.join(positive_folder, filename)
    if os.path.isfile(file_path):
        mrn = filename.split(".")[0]
        first_dash_index = mrn.find('-')
        desired_substring = mrn[first_dash_index + 1:]
        if desired_substring in new_other_set:
            output_path = os.path.join(neg_output_folder, filename)
            shutil.copyfile(file_path, output_path)
            count_reinke += 1
print('number of existed Reinke Edema:', count_reinke)  
total_files_copied = len(os.listdir(neg_output_folder))
print(f"Total samples copied: {total_files_copied}")
#===================================================================
#finding Reinke Edema in Diagnosis2 when type_of_mass is null and adding it to negative folder        

data2 = pd.read_csv('C:/Users/zahra/VoiceColab - Copy/dataset/metadata/data.csv', usecols=['MRN', 'Diagnosis', 'Diagnosis2','Type_of_Mass'])
new_other_2 = data2[(data2['Type_of_Mass'].isnull()) & (data2['Diagnosis2'] == "Reinke's Edema")].copy()        
new_other_set2 = set(new_other_2['MRN'])
count_reinke = 0

for filename in os.listdir(positive_folder):
    file_path = os.path.join(positive_folder, filename)
    if os.path.isfile(file_path):
        mrn = filename.split(".")[0]
        first_dash_index = mrn.find('-')
        desired_substring = mrn[first_dash_index + 1:]
        if desired_substring in new_other_set2:
            output_path = os.path.join(neg_output_folder, filename)
            shutil.copyfile(file_path, output_path)
            count_reinke += 1
print('number of existed Reinke Edema:', count_reinke)  
total_files_copied = len(os.listdir(neg_output_folder))
print(f"Total samples copied: {total_files_copied}")
#====================================================================
# To remove noisy sample in negative data
file_to_remove = '1-U3188683-8'
file_path = os.path.join(neg_output_folder, file_to_remove)
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File '{file_to_remove}' has been removed.")
else:
    print(f"File '{file_to_remove}' does not exist in the negative folder.")