'''This code plays both original audios and the corresponding augmented audios 
'''
from src.make_dataset_folder import make_dataset_folder
from src.get_dataset_name import get_dataset_name
from config.config import (folder_path_test, folder_path_train,
                     helper_output_path)


import os
import pydub
import random
import pickle
import tempfile
import numpy as np
import simpleaudio as sa
from IPython.display import Audio, display

#%% 
def play_samples(folder_path, filename1, filename2):
    
    dataset_name = get_dataset_name(folder_path, -5)
    learning_selection = input("Enter 'c' to navigate to CNN folder or 't' for Transfer-Learning: ")
    if learning_selection.lower() == 'c':
        dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='CNN')
        
    elif learning_selection.lower() == 't':
        transfer_learning = input("Enter 'r' for Resnet50 or 'i' for InceptionV3 or 'x' for Xception: ")
        if transfer_learning.lower() == 'r':
            dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='Resnet50')
        if transfer_learning.lower() == 'i':
            dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='InceptionV3')
        if transfer_learning.lower() == 'x':
            dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='Xception')
        else:
            print("WRONG KEY!!!")    
    
    num_samples = int(input("Enter the number of samples to play: "))
    variable_name = "train" if "train" in filename1 else "validation" if "validation" in filename1 else "test"
    if "train" in variable_name:
         print("Data type: train")
    if "validation" in variable_name:
         print("Data type: validation")
    if "test" in variable_name:
         print("Data type: test")
    print('==================================================') 

    output_file_path1 = os.path.join(dataset_folder_helper,filename1)
    with open(output_file_path1, "rb") as file1:
         augmented_train = pickle.load(file1)
         DATA = augmented_train 

    output_file_path = os.path.join(dataset_folder_helper,filename2)
    with open(output_file_path, "rb") as file:
         balanced_train_data = pickle.load(file)
         original_data = balanced_train_data 

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
            audio_segment_aug = pydub.AudioSegment(signal_mono.tobytes(), frame_rate=sample_rate, sample_width=signal_mono.dtype.itemsize, channels=1)
            temp_dir_aug = tempfile.mkdtemp()
            temp_file_path_aug = os.path.join(temp_dir_aug, "temp_audio.wav")
            audio_segment_aug.export(temp_file_path_aug, format="wav")
            wave_obj_aug = sa.WaveObject.from_wave_file(temp_file_path_aug)
            play_obj_aug = wave_obj_aug.play()
            play_obj_aug.wait_done()
            
            original_sample = next((sample1 for sample1 in original_data if sample1[1] == name), None)
            if original_sample:
                original_signal = np.array(original_sample[0])
                original_signal_mono = original_signal.astype(np.float32)
                print(f"Sample {i+1} - Original - Name: {original_sample[1]}\tLabel: {original_sample[2]}")
                audio_segment = pydub.AudioSegment(original_signal_mono.tobytes(), frame_rate=sample_rate, sample_width=signal_mono.dtype.itemsize, channels=1)
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
            

if __name__ == '__main__':

   sample_rate = 44100
   folder_path_train = folder_path_train
   folder_path_test = folder_path_test

   environment = input("If you are running the code in Colab, enter 'c'. If you are running the code in Spyder, enter 's': ")

   play_samples(folder_path_train, 'augmented_train.pkl', 'balanced_train_data.pkl' )
   #play_samples(folder_path_test, 'augmented_test.pkl', 'balanced_test_data.pkl' )

