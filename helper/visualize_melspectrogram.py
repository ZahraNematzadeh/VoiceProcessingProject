'''
This code visualize the Melspectrograms.
First run the main code to create the melspect_data.pkl file, 
then read the pkl file for visualization.
'''

from src.make_dataset_folder import make_dataset_folder
from src.get_dataset_name import get_dataset_name
from config.config import (folder_path_test, folder_path_train,
                     helper_output_path, plots_output_path, sample_rate)

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def visualizing_melspect(folder_path, filename):
    
    dataset_name = get_dataset_name(folder_path, -5)
    learning_selection = input("Enter 'c' for CNN or 't' for Transfer-Learning: ")
        
    if learning_selection.lower() == 'c':
        dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='CNN')
        dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Melspectrogram', learning_name= None)
        
    elif learning_selection.lower() == 't':
        transfer_learning = input("Enter 'r' for Resnet50 or 'i' for InceptionV3 or 'x' for Xception: ")
        if transfer_learning.lower() == 'r':
            dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='Resnet50')
            dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Melspectrogram', learning_name= None)
        if transfer_learning.lower() == 'i':
            dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='InceptionV3')
            dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Melspectrogram', learning_name= None)
        if transfer_learning.lower() == 'x':
            dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='Xception')
            dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Melspectrogram', learning_name= None)
    else:
        print("WRONG KEY!!!")    
            
    num_samples = int(input("Enter the number of samples to visualize: "))
    variable_name = "train" if "train" in filename else "validation" if "validation" in filename else "test"
    if "train" in variable_name:
        print("Data type: train")
    if "validation" in variable_name:
        print("Data type: validation")
    if "test" in variable_name:
        print("Data type: test")
    print('==============================================================')

    
    output_file_path = os.path.join(dataset_folder_helper,filename )
    with open(output_file_path, "rb") as file:
        melspect_data = pickle.load(file)
    
    for i in range(num_samples):
        
        random.shuffle(melspect_data)
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
        
        plt.savefig(os.path.join(dataset_folder_plots,  f'Melspectrogram_{i}.png'), dpi =1000)
        plt.clf()
    plt.close()

  
if __name__ == '__main__':
    
    sample_rate = sample_rate
    folder_path_train = folder_path_train
    folder_path_test = folder_path_test
    
    visualizing_melspect(folder_path_train, 'melspect_train_data.pkl')
  # visualizing_melspect(folder_path_test, 'melspect_test_data.pkl')




