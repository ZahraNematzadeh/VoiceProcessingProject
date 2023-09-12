import os
import librosa
import numpy as np
from scipy.io import wavfile

e_folder_path = 'C:/Users/zahra/VoiceColab/dataset/RawDataGCloud/1_Sustained_e/eee/'
ah_folder_path = 'C:/Users/zahra/VoiceColab/dataset/2_ah/new_ah/raw_ah/'
sissy_folder_path = 'C:/Users/zahra/VoiceColab/dataset/8_sissy/new_sissy/raw_sissy/'

output_directory = 'C:/Users/zahra/VoiceColab/dataset/ConcatenatedAudio/e_ah_sissy/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

audio_data_dict = {}
sample_rates = {}
matching_filenames_e = {}
matching_filenames_ah = {}
matching_filenames_sissy = {}

#Finding identifiers in e 
for filename in os.listdir(e_folder_path):
    if filename.endswith(".wav"):
        parts = filename.split('-', 1)
        identifier = parts[1]  
        if identifier not in matching_filenames_e:
            matching_filenames_e[identifier] = []
        matching_filenames_e[identifier].append(os.path.join(e_folder_path, filename))

#Finding identifiers in ah
for filename in os.listdir(ah_folder_path):
    if filename.endswith(".wav"):
        identifier = filename.split('.')[0]
        if identifier not in matching_filenames_ah:
            matching_filenames_ah[identifier] = []
        matching_filenames_ah[identifier].append(os.path.join(ah_folder_path, filename))

# Finding identifiers in 'sissy'
for filename in os.listdir(sissy_folder_path):
    if filename.endswith(".wav"):
        identifier = filename.split('.')[0]
        if identifier not in matching_filenames_sissy:
            matching_filenames_sissy[identifier] = []
        matching_filenames_sissy[identifier].append(os.path.join(sissy_folder_path, filename))


# Iterate over identifiers in ah
audio_not_found_count = 0
sissy_audio_not_found_count = 0
e_identifiers = [os.path.splitext(os.path.basename(key))[0] for key in matching_filenames_e.keys()]

for ah_identifier, ah_audio_paths in matching_filenames_ah.items():
    if (ah_identifier + '.wav') in matching_filenames_e.keys():
        if ah_identifier in matching_filenames_sissy.keys():
            e_file_path = matching_filenames_e[ah_identifier + '.wav'][0]
            e_audio, sr = librosa.load(e_file_path, sr=None)
            sissy_file_path = matching_filenames_sissy[ah_identifier][0]
            sissy_audio, sr = librosa.load(sissy_file_path, sr=None)

            concatenated_audio = e_audio  
            for ah_audio_path in ah_audio_paths:
                if os.path.exists(ah_audio_path):
                    ah_audio, sr = librosa.load(ah_audio_path, sr=None)
                    concatenated_audio = np.concatenate((concatenated_audio, ah_audio), axis=None)
            
            concatenated_audio = np.concatenate((concatenated_audio, sissy_audio), axis=None)  # Add 'sissy' audio

            output_filename = f"{ah_identifier}.wav"
            output_path = os.path.join(output_directory, output_filename)
            wavfile.write(output_path, sr, concatenated_audio)
        else:
            audio_not_found_count += 1
    else:
        audio_not_found_count += 1
    
print(f"Concatenation completed. {audio_not_found_count} 'ah' audio files were not found in 'e' folder.")
print('########################################################################################')
#print(f"Concatenation completed. {sissy_audio_not_found_count} 'sissy' audio files were not found in 'ah' folder.")














