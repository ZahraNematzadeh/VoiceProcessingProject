import os 

def make_dataset_folder(folder_path, dataset_name, visualizing, learning_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    dataset_folder = os.path.join(folder_path, dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        
    visualizing_folder = os.path.join(dataset_folder, visualizing)
    if not os.path.exists(visualizing_folder):
        os.makedirs(visualizing_folder)
    
    
    if learning_name is not None:
        learning_folder = os.path.join(visualizing_folder, learning_name)
        if not os.path.exists(learning_folder):
            os.makedirs(learning_folder)
        return learning_folder
    else:
        return visualizing_folder 