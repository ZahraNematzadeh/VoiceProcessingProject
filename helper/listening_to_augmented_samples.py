import os
import pydub
import random
import pickle
import tempfile
import numpy as np
import simpleaudio as sa
from IPython.display import Audio, display

with open('outputs/CodeFiles/HelpersOutputs/augmented_train.pkl', "rb") as file:
    augmented_train = pickle.load(file)
    
with open('outputs/CodeFiles/HelpersOutputs/augmented_test.pkl', "rb") as file:
    augmented_test = pickle.load(file)

with open('outputs/CodeFiles/HelpersOutputs/balanced_train_data.pkl', "rb") as file:
    balanced_train_data = pickle.load(file)
    
with open('outputs/CodeFiles/HelpersOutputs/balanced_test_data.pkl', "rb") as file:
    balanced_test_data = pickle.load(file)

environment = input("If you are running the code in Colab, enter 'c'. If you are running the code in Spyder, enter 's': ")

#%% 
DATA = augmented_train                  #or augmented_test  
original_data = balanced_train_data     #or original_test
sample_rate = 44100
num_samples = int(input("Enter the number of samples to listen to: "))
random.shuffle(DATA)

selected_samples = [sample for sample in DATA if '_' in sample[1]]
selected_samples = selected_samples[:num_samples]
for i, (signal, name, label) in enumerate(selected_samples):
    signal = np.array(signal)
    signal_mono = signal.astype(np.float32)
    print(f"Sample {i+1} - Augmented - Name: {name}\tLabel: {label}")
    if environment.lower() == 'c':
        audio_widget = Audio(data=signal_mono, rate=sample_rate, autoplay=True)
        display(audio_widget)
        original_sample = next((sample1 for sample1 in original_data if sample1[1] == name), None)
        if original_sample:
            original_signal = np.array(original_sample[0])
            original_signal_mono = original_signal.astype(np.float32)
            print(f"Sample {i+1} - Original - Name: {original_sample[1]}\tLabel: {original_sample[2]}")
            audio_widget_original = Audio(data=original_signal_mono, rate=sample_rate, autoplay=True)
            display(audio_widget_original)

        print("=================================================")
        print("=================================================")

    elif environment.lower() == 's':
        audio_segment = pydub.AudioSegment(signal_mono.tobytes(), frame_rate=sample_rate, sample_width=signal_mono.dtype.itemsize, channels=1)
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_audio.wav")
        audio_segment.export(temp_file_path, format="wav")
        wave_obj = sa.WaveObject.from_wave_file(temp_file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        original_sample = next((sample1 for sample1 in original_data if sample1[1] == name), None)
        if original_sample:
            original_signal = np.array(original_sample[0])
            original_signal_mono = original_signal.astype(np.float32)
            print(f"Sample {i+1} - Original - Name: {original_sample[1]}\tLabel: {original_sample[2]}")
            audio_segment = pydub.AudioSegment(signal_mono.tobytes(), frame_rate=sample_rate, sample_width=signal_mono.dtype.itemsize, channels=1)
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, "temp_audio.wav")
            audio_segment.export(temp_file_path, format="wav")
            wave_obj = sa.WaveObject.from_wave_file(temp_file_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()

        print("=================================================")
        print("=================================================")
    else:
        print("Invalid input. Please enter 'c' for Colab or 's' for Spyder.")