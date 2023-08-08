'''
This code visualize the leaf images.
First run the main code to create the leaf_data.pkl file, 
then read the pkl file for visualization.
'''
import sys
sys.path.append('/content/VoicePathologyDetection')

from src.make_dataset_folder import make_dataset_folder
from src.get_dataset_name import get_dataset_name
from config.config import (folder_path_test, folder_path_train,
                     helper_output_path, plots_output_path)

import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt


def visualizing_leaf(folder_path, filename):
    
    dataset_name = get_dataset_name(folder_path, -4)
    learning_selection = input("Enter 'c' to navigate to CNN folder or 't' for Transfer-Learning: ")
    if learning_selection.lower() == 'c':
        dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Leaf', learning_name ='CNN')
        dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Leaf', learning_name = None)
        
    elif learning_selection.lower() == 't':
        transfer_learning = input("Enter 'r' for Resnet50 or 'i' for InceptionV3 or 'x' for Xception: ")
        if transfer_learning.lower() == 'r':
            dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Leaf', learning_name ='Resnet50')
            dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Leaf', learning_name = None)
        if transfer_learning.lower() == 'i':
            dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Leaf', learning_name ='InceptionV3')
            dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Leaf', learning_name = None)
        if transfer_learning.lower() == 'x':
            dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Leaf', learning_name ='Xception')
            dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Leaf', learning_name = None)
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
        leaf_data = pickle.load(file)     
         
    for i in range(num_samples):
        random.shuffle(leaf_data)
        leaf_array, filename, label = leaf_data[i]
        leaf_array = np.array(leaf_array)
        if len(leaf_array.shape) > 2:
            leaf_array = np.squeeze(leaf_array)
        
        print("Leaf shape:", leaf_array.shape)
        plt.figure(figsize=(4, 3))
        plt.title(f'{filename} | Label: {label}', fontsize=10)
        plt.pcolormesh(leaf_array)
        plt.savefig(os.path.join(dataset_folder_plots,  f'leaf_{i}.png'), dpi =1000)
        plt.clf()
    plt.close()
         
     
if __name__ == '__main__':
    folder_path_train = folder_path_train
    folder_path_test = folder_path_test
    
    visualizing_leaf(folder_path_train, "leaf_train_data.pkl")
 # visualizing_leaf(folder_path_test, "leaf_test_data.pkl")