'''
This code visualize the melspectrogram and corresponding leaf images.
First run the main code to create the leaf_data.pkl  and melspect_data.pkl, 
then read the both pkl file for visualization.
'''

from src.make_dataset_folder import make_dataset_folder
from src.get_dataset_name import get_dataset_name
from config import (folder_path_test, folder_path_train,
                     helper_output_path, plots_output_path)

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def visualizing_melspect_and_leaf(folder_path, mel_filename, leaf_filename):
    
    dataset_name = get_dataset_name(folder_path, -5)
    learning_selection = input("Enter 'c' for CNN or 't' for Transfer-Learning: ")
    if learning_selection.lower() == 'c':
       
        mel_dataset_folder_helper = make_dataset_folder(helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='CNN')
        mel_dataset_folder_plots = make_dataset_folder(plots_output_path, dataset_name, visualizing='Melspectrogram', learning_name=None)
        
        leaf_dataset_folder_helper = make_dataset_folder(helper_output_path, dataset_name, visualizing='Leaf', learning_name='CNN')
        leaf_dataset_folder_plots = make_dataset_folder(plots_output_path, dataset_name, visualizing='Leaf', learning_name=None)
        
    elif learning_selection.lower() == 't':
        transfer_learning = input("Enter 'r' for Resnet50 or 'i' for InceptionV3 or 'x' for Xception: ")
        if transfer_learning.lower() in ['r', 'i', 'x']:
            mel_dataset_folder_helper = make_dataset_folder(helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name=transfer_learning.capitalize())
            mel_dataset_folder_plots = make_dataset_folder(plots_output_path, dataset_name, visualizing='Melspectrogram', learning_name=None)
            leaf_dataset_folder_helper = make_dataset_folder(helper_output_path, dataset_name, visualizing='Leaf', learning_name=transfer_learning.capitalize())
            leaf_dataset_folder_plots = make_dataset_folder(plots_output_path, dataset_name, visualizing='Leaf', learning_name=None)
        else:
            print("WRONG KEY!!!")
    else:
        print("WRONG KEY!!!")
            
    num_samples = int(input("Enter the number of samples to visualize: "))
    variable_name = "train" if "train" in mel_filename else "validation" if "validation" in mel_filename else "test"
    
    if "train" in variable_name:
        print("Data type: train")
    elif "validation" in variable_name:
        print("Data type: validation")
    elif "test" in variable_name:
        print("Data type: test")
    
    print('==============================================================')

    mel_output_file_path = os.path.join(mel_dataset_folder_helper, mel_filename)
    leaf_output_file_path = os.path.join(leaf_dataset_folder_helper, leaf_filename)
    
    with open(mel_output_file_path, "rb") as mel_file, open(leaf_output_file_path, "rb") as leaf_file:
        melspect_data = pickle.load(mel_file)
        leaf_data = pickle.load(leaf_file)
        
    for i in range(num_samples):
        
        random.shuffle(melspect_data)
        mel_array, mel_filename, mel_label = melspect_data[i]

        if len(mel_array.shape) > 2:
            mel_array = np.squeeze(mel_array)
            
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        librosa.display.specshow(mel_array, sr=sample_rate, x_axis='time', y_axis='mel', fmax=5000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Melspectrogram - {mel_filename} | Label: {mel_label}', fontsize=10)
        
        corresponding_leaf = next((leaf for leaf in leaf_data if leaf[1] == os.path.splitext(mel_filename)[0]), None)
        
        if corresponding_leaf:
            leaf_array, leaf_filename, leaf_label = corresponding_leaf
            leaf_array = np.array(leaf_array)
            if len(leaf_array.shape) > 2:
                leaf_array = np.squeeze(leaf_array)

            print("Melspectrogram:", mel_filename)
            print("Leaf:", leaf_filename)

            plt.subplot(1, 2, 2)
            plt.pcolormesh(leaf_array)
            plt.colorbar()
            plt.title(f'Leaf - {leaf_filename} | Label: {leaf_label}', fontsize=10)
        
            plt.tight_layout()
            plt.savefig(os.path.join(mel_dataset_folder_plots, f'Combined_{i}.png'), dpi=1000)
            plt.clf()
            
        else:
            print("No corresponding leaf data found for", mel_filename)
        
    plt.close()
  
if __name__ == '__main__':
    sample_rate = 44100
    folder_path_train = folder_path_train
    folder_path_test = folder_path_test
    
    visualizing_melspect_and_leaf(folder_path_train, 'melspect_train_data.pkl', 'leaf_train_data.pkl')
    # visualizing_melspect_and_leaf(fold