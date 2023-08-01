import os
import librosa
import numpy as np

#%%
folder_path_train= 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/2_DataLabeling/CleanedData/Test_Train/train'
folder_path_test= 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/2_DataLabeling/CleanedData/Test_Train/val'
padded_train = []
padded_test = []

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
                padded_audio = padding_audio(y, 5, 44100)
                label = os.path.basename(root)
                padded_list.append((padded_audio, filename, label))

if __name__=='__main__':
    pad_audio(folder_path_train, padded_train)
    pad_audio(folder_path_test, padded_test)
    
    #Inspecting padded audios
    for item in padded_train:    #or padded_test
        print(f"Signal: {item[0]}\tName: {item[1]}\tLabel: {item[2]}")
        
    #Inspecting min and max values in audios
    padded_train_array = np.array([item[0] for item in padded_train]) #or padded_test
    max_value = np.max(padded_train_array)
    min_value = np.min(padded_train_array)
    print("Maximum value:", max_value)
    print("Minimum value:", min_value)