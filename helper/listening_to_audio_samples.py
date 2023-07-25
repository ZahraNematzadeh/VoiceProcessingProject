'''This code shows audios after padding and oversampling 
'''
import os
import pydub
import pickle
import random
import tempfile
import numpy as np
import simpleaudio as sa
from IPython.display import Audio, display

with open('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs/balanced_train_data.pkl', "rb") as file:
    balanced_train_data = pickle.load(file)
    
with open('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs/balanced_test_data.pkl', "rb") as file:
    balanced_test_data = pickle.load(file)
    
environment = input("If you are running the code in Colab, enter 'c'. If you are running the code in Spyder, enter 's': ")

#%%
def listen_to_audio(balanced_data):
    DATA = balanced_data      
    sample_rate = 44100
    num_samples = int(input("Enter the number of samples to listen to: "))
    variable_name = [name for name, value in globals().items() if value is DATA][0]
    if "train" in variable_name:
        print("Data type: train")
    if "validation" in variable_name:
        print("Data type: validation")
    if "test" in variable_name:
        print("Data type: test")
    print('==================================================') 
    random.shuffle(DATA)
    for i in range(num_samples):
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


listen_to_audio(balanced_train_data)
#listen_to_audio(balanced_test_data)





    
    
    
    
    
