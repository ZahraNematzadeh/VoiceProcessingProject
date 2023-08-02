'''This code shows audios after padding and oversampling 
'''
from src.make_dataset_folder import make_dataset_folder
from src.get_dataset_name import get_dataset_name
from config import (folder_path_test, folder_path_train,
                     helper_output_path, plots_output_path)

import os
import pydub
import pickle
import random
import tempfile
import numpy as np
import simpleaudio as sa
from IPython.display import Audio, display
#%%
def play_audio(folder_path,filename):
    
    sample_rate = 44100
    
    dataset_name = get_dataset_name(folder_path, -5)
    visualizing_selection = input("Enter 'y' if you would like to play audios: ")
    if visualizing_selection.lower() == 'y':
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

    num_samples = int(input("Enter number of audios to play: "))
    variable_name = "train" if "train" in filename else "validation" if "validation" in filename else "test"
    if "train" in variable_name:
        print("Data type: train")
    if "validation" in variable_name:
        print("Data type: validation")
    if "test" in variable_name:
        print("Data type: test")
    print('==================================================') 
    
    output_file_path = os.path.join(dataset_folder_helper,filename )
    with open(output_file_path, "rb") as file:
        balanced_data = pickle.load(file)
        DATA = balanced_data 
    
    for i in range(num_samples):
        random.shuffle(DATA)

        signal = DATA[i][0]
        name = DATA[i][1]
        label = DATA[i][2]
        signal_mono = signal.astype(np.float32)
        print(f"Sample {i+1} - Name: {name}\tLabel: {label}")
        if environment.lower() == 'c':
            audio_widget = Audio(data=signal_mono, rate=sample_rate, autoplay=True)
            display(audio_widget)
        elif environment.lower() == 's':
            audio_segment = pydub.AudioSegment(signal_mono.tobytes(), frame_rate=sample_rate, sample_width=signal_mono.dtype.itemsize, channels=1)
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, "temp_audio.wav")
            audio_segment.export(temp_file_path, format="wav")
            wave_obj = sa.WaveObject.from_wave_file(temp_file_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        else:
            print("Invalid input. Please enter 'c' for Colab or 's' for Spyder.")


if __name__ == '__main__':
    
   folder_path_train = folder_path_train
   folder_path_test = folder_path_test
   
   environment = input("If you are running the code in Colab, enter 'c'. If you are running the code in Spyder, enter 's': ")
    
   #play_audio(folder_path_train,'balanced_train_data.pkl' )
   play_audio(folder_path_test,'balanced_test_data.pkl' )






    
    
    
    
    
