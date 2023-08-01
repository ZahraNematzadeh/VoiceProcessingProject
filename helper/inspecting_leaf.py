'''
This code visualize the leaf images.
First run the main code to create the leaf_data.pkl file, 
then read the pkl file for visualizing.
'''
from src.make_dataset_folder import make_dataset_folder
from src.get_dataset_name import get_dataset_name

import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt


def visualizing_leaf(folder_path, filename):
    
    dataset_name = get_dataset_name(folder_path, -5)
    visualizing_selection = input("Enter 'y' if you like to visualize leaf images: ")
    
    if visualizing_selection.lower() == 'y':
        learning_selection = input("Enter 'c' for CNN or 't' for Transfer-Learning: ")
        
        if learning_selection.lower() == 'c':
            dataset_folder_helper = make_dataset_folder ('C:/Users/zahra/VoiceColab/outputs/HelpersOutputs', dataset_name, visualizing='Leaf', learning_name='CNN')
            dataset_folder_plots = make_dataset_folder ('C:/Users/zahra/VoiceColab/outputs/Plots', dataset_name, visualizing='Leaf', learning_name='CNN')
        
        elif learning_selection.lower() == 't':
            transfer_learning = input("Enter 'r' for Resnet50 or 'i' for InceptionV3 or 'x' for Xception: ")
            if transfer_learning.lower() == 'r':
                dataset_folder_helper = make_dataset_folder ('C:/Users/zahra/VoiceColab/outputs/HelpersOutputs', dataset_name, visualizing='Leaf', learning_name='Resnet50')
                dataset_folder_plots = make_dataset_folder ('C:/Users/zahra/VoiceColab/outputs/Plots', dataset_name, visualizing='Leaf', learning_name='Resnet50')
            if transfer_learning.lower() == 'i':
                dataset_folder_helper = make_dataset_folder ('C:/Users/zahra/VoiceColab/outputs/HelpersOutputs', dataset_name, visualizing='Leaf', learning_name='InceptionV3')
                dataset_folder_plots = make_dataset_folder ('C:/Users/zahra/VoiceColab/outputs/Plots', dataset_name, visualizing='Leaf', learning_name='InceptionV3')
            if transfer_learning.lower() == 'x':
                dataset_folder_helper = make_dataset_folder ('C:/Users/zahra/VoiceColab/outputs/HelpersOutputs', dataset_name, visualizing='Leaf', learning_name='Xception')
                dataset_folder_plots = make_dataset_folder ('C:/Users/zahra/VoiceColab/outputs/Plots', dataset_name, visualizing='Leaf', learning_name='Xception')
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
    folder_path_train= 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/train'
    folder_path_test= 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/val'
    
    visualizing_leaf(folder_path_train, "augmented_train.pkl")
 # visualizing_leaf(folder_path_test, "augmented_test.pkl")
