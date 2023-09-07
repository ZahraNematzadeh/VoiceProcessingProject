import os
import librosa
import numpy as np
from scipy.io import wavfile

e_folder_path = 'C:/Users/zahra/VoiceColab/dataset/RawDataGCloud/1_Sustained_e/eee/'
oh_folder_path = 'C:/Users/zahra/VoiceColab/dataset/RawDataGCloud/3_Sustained_oh/'
output_directory = 'C:/Users/zahra/VoiceColab/dataset/ConcatenatedAudio/e_oh/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

audio_data_dict = {}
sample_rates = {}
matching_filenames_e = {}
matching_filenames_oh = {}

#Finding identifiers in e 
for filename in os.listdir(e_folder_path):
    if filename.endswith(".wav"):
        parts = filename.split('-', 1)
        identifier = parts[1]  
        if identifier not in matching_filenames_e:
            matching_filenames_e[identifier] = []
        matching_filenames_e[identifier].append(os.path.join(e_folder_path, filename))

#Finding identifiers in oh
for filename in os.listdir(oh_folder_path):
    if filename.endswith(".wav"):
        identifier = filename.split('.')[0]
        if identifier not in matching_filenames_oh:
            matching_filenames_oh[identifier] = []
        matching_filenames_oh[identifier].append(os.path.join(oh_folder_path, filename))

# Iterate over identifiers in oh
oh_audio_not_found_count = 0
e_identifiers = [os.path.splitext(os.path.basename(key))[0] for key in matching_filenames_e.keys()]

for oh_identifier, oh_audio_paths in matching_filenames_oh.items():
    if (oh_identifier +'.wav') in matching_filenames_e.keys():
        e_file_path = matching_filenames_e[ oh_identifier +'.wav'][0]
        e_audio, sr = librosa.load(e_file_path, sr=None)
        concatenated_audio = e_audio
    else:
        oh_audio_not_found_count += 1 
    
    for oh_audio_path in oh_audio_paths:
        if os.path.exists(oh_audio_path):
            oh_audio, sr = librosa.load(oh_audio_path, sr=None)
            concatenated_audio = np.concatenate((concatenated_audio, oh_audio), axis=None)  
        
        output_filename = f"{oh_identifier}.wav"
        output_path = os.path.join(output_directory, output_filename)
        wavfile.write(output_path, sr, concatenated_audio)
    
print(f"Concatenation completed. {oh_audio_not_found_count} 'oh' audio files were not found in 'e' folder.")















