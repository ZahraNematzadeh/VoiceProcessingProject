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
    dataset_folder_helper = make_dataset_folder ('C:/Users/zahra/VoiceColab/outputs/HelpersOutputs', dataset_name)
    dataset_folder_plots = make_dataset_folder ('C:/Users/zahra/VoiceColab/outputs/Plots', dataset_name)

    output_file_path = os.path.join(dataset_folder_helper,filename )
    with open(output_file_path, "rb") as file:
         leaf_data = pickle.load(file)
    
    num_samples = int(input("Enter the number of samples to visualize: "))
    variable_name = "train" if "train" in filename else "validation" if "validation" in filename else "test"
    if "train" in variable_name:
        print("Data type: train")
    if "validation" in variable_name:
        print("Data type: validation")
    if "test" in variable_name:
        print("Data type: test")
    print('==============================================================')
    random.shuffle(leaf_data)
   
    
    for i in range(num_samples):
        leaf_array, filename, label = leaf_data[i]

        if len(leaf_array.shape) > 2:
            leaf_array = np.squeeze(leaf_array)

        print("Leaf shape:", leaf_array.shape)
        plt.title(f'{filename} | Label: {label}', fontsize=10)
        plt.pcolormesh(leaf_array)
        plt.savefig(os.path.join(dataset_folder_plots,  f'leaf_{i}.png'), dpi =1000)
        plt.clf()
    plt.close()
     
     
if __name__ == '__main__':
    folder_path_train= 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/train'
    folder_path_test= 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/val'
    
    visualizing_leaf(folder_path_train, "leaf_train_data.pkl")
 # visualizing_leaf(folder_path_test, "leaf_test_data.pkl")
