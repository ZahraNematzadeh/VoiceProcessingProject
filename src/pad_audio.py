import os
import librosa
import numpy as np
from config.config import (sample_rate, max_duration)


#%%
def pad_audio(folder_path, sr, max_duration):
    padded_list = []
    num_expected_samples = int(sr * max_duration)
        
    for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith('.wav'):
                    file_path = os.path.join(root, filename)
                    label = os.path.basename(root)
                    y, _ = librosa.load(file_path, sr=None, mono=True)
                    #audio_resampled = librosa.resample(y,orig_sr= 44100, target_sr=48000)
                    signal = np.array(y)
                    if len(signal) < num_expected_samples:
                        num_missing_items = num_expected_samples - len(signal)
                        padded_signal = np.pad(signal, (0, num_missing_items), mode="constant")
                    elif len(signal) > num_expected_samples:
                        padded_signal = signal[:num_expected_samples]
                    elif len(signal) == num_expected_samples:
                        padded_signal = signal
                 
                    padded_list.append((padded_signal, filename, label))
           
    return padded_list
                

if __name__=='__main__':
        
    folder_path_train= 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/train'
    folder_path_test= 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/val'

    padded_train_1 = pad_audio(folder_path_train, sample_rate, max_duration)
    padded_test_1 = pad_audio(folder_path_test, sample_rate, max_duration)
    
    #Inspecting padded audios
    for item in padded_train_1:    #or padded_test
        print(f"Signal: {item[0]}\tName: {item[1]}\tLabel: {item[2]}")
        
    #Inspecting min and max values in audios
    padded_train_array = np.array([item[0] for item in padded_train_1]) #or padded_test
    max_value = np.max(padded_train_array)
    min_value = np.min(padded_train_array)
    print("Maximum value:", max_value)
    print("Minimum value:", min_value)