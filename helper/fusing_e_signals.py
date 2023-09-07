import os
import librosa
import numpy as np
from scipy.io import wavfile

folder_path = 'C:/Users/zahra/VoiceColab/dataset/RawDataGCloud/1_Sustained_e/eee/'
output_directory = 'C:/Users/zahra/VoiceColab/dataset/1_e/test_train/ConcatenatedAudio/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

audio_data_dict = {}
sample_rates = {}
matching_filenames = {}
pattern = ["1-", "2-", "3-"]

all_audio = os.listdir(folder_path)

for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        parts = filename.split('-', 1)
        identifier = parts[1]  
        if identifier not in matching_filenames:
            matching_filenames[identifier] = []
        matching_filenames[identifier].append(filename)
 
for identifier, filenames in matching_filenames.items():
    if len(filenames) == 3:
        audio_data_dict[identifier] = filenames
        
#for identifier, filenames in audio_data_dict.items():
   # print(f"Identifier: {identifier}, Audio Files: {filenames}")  

for identifier, filenames in audio_data_dict.items():
    concatenated_audio = None
    for i in range(1, 4):  # Iterate over '1-filename', '2-filename', '3-filename'
        filename = f"{i}-{identifier}"
        audio_path = os.path.join(folder_path, filename)
        audio, sr = librosa.load(audio_path, sr=None)

        if concatenated_audio is None:
            concatenated_audio = audio
        else:
            concatenated_audio = np.concatenate((concatenated_audio, audio), axis=None)

    output_filename = f"{identifier}"
    output_path = os.path.join(output_directory, output_filename)
    wavfile.write(output_path, sr, concatenated_audio)

print("Concatenation completed.")

#%%
#Inspecting how many corresponding samples are there in the folder
identifier_counts = {}
identifiers_with_count_three = 0
identifiers_with_count_two = 2
identifiers_with_count_one = 1

for identifier, filenames in matching_filenames.items():
    count = len(filenames)
    identifier_counts[identifier] = count
    if count == 3:
        identifiers_with_count_three += 1
    elif count == 2:
        identifiers_with_count_two += 1
    elif count == 1:
        identifiers_with_count_one += 1
print(" identifiers_with_count_three",  identifiers_with_count_three,
      " identifiers_with_count_two",  identifiers_with_count_two,
      " identifiers_with_count_one",  identifiers_with_count_one)
