import os
import librosa
import numpy as np

#%%
def padding_audio(signal, max_duration, sample_rate):
    padded_signal = np.array(signal)
    num_expected_samples = int(sample_rate * max_duration)
    if len(padded_signal) < num_expected_samples:
        num_missing_items = num_expected_samples - len(padded_signal)
        padded_signal = np.pad(padded_signal, (0, num_missing_items), mode="constant")
    elif len(padded_signal) > num_expected_samples:
        padded_signal = padded_signal[:num_expected_samples]
    elif len(padded_signal) == num_expected_samples:
        padded_signal = padded_signal
    return padded_signal

def pad_audio(folder_path, padded_list):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                y, sr = librosa.load(file_path, sr=None, mono=True)
                audio_resampled = librosa.resample(y,orig_sr= 44100, target_sr=48000)
                padded_audio = padding_audio(audio_resampled , 5, 48000)
                label = os.path.basename(root)
                padded_list.append((padded_audio, filename, label))

    return padded_list

if __name__=='__main__':
    
    padded_train = []
    padded_test = []
    
    folder_path_train= 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/train'
    folder_path_test= 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/val'

    padded_train_1 = pad_audio(folder_path_train, padded_train)
    padded_test_1 = pad_audio(folder_path_test, padded_test)
    
    #Inspecting padded audios
    for item in padded_train_1:    #or padded_test
        print(f"Signal: {item[0]}\tName: {item[1]}\tLabel: {item[2]}")
        
    #Inspecting min and max values in audios
    padded_train_array = np.array([item[0] for item in padded_train_1]) #or padded_test
    max_value = np.max(padded_train_array)
    min_value = np.min(padded_train_array)
    print("Maximum value:", max_value)
    print("Minimum value:", min_value)