import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


sample_rate = 44100

with open('C:/Users/zahra/VoiceColab/outputs/CodeFiles/HelpersOutputs/melspect_train_data.pkl', "rb") as file:
    melspect_train_data = pickle.load(file)
    
with open('C:/Users/zahra/VoiceColab/outputs/CodeFiles/HelpersOutputs/melspect_train_data.pkl', "rb") as file:
    melspect_test_data = pickle.load(file)

def visualizing_melspect(melspect_data):
    num_samples = int(input("Enter the number of samples to visualize: "))
    variable_name = [name for name, value in globals().items() if value is melspect_data][0]
    if "train" in variable_name:
        print("Data type: train")
    if "validation" in variable_name:
        print("Data type: validation")
    if "test" in variable_name:
        print("Data type: test")
    print('==============================================================')

    random.shuffle(melspect_data)
    for i in range(num_samples):
        melspect_array, filename, label = melspect_data[i]

        if len(melspect_array.shape) > 2:
            melspect_array = np.squeeze(melspect_array)

        print("Melspectrogram shape:", melspect_array.shape)
        plt.figure(figsize=(4.4, 2.8))
        librosa.display.specshow(melspect_array, sr=sample_rate, x_axis='time', y_axis='mel',fmax=5000)
        cbar = plt.colorbar(format='%+2.0f dB')
        cbar.ax.tick_params(labelsize=8)
        plt.title(f'{filename} | Label: {label}', fontsize=10)
        plt.tight_layout()
        plt.show()

visualizing_melspect(melspect_train_data)
#visualizing_melspect(melspect_test_data)