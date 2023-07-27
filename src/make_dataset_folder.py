import os 

def make_dataset_folder(folder_path, dataset_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    dataset_folder = os.path.join(folder_path, dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    return dataset_folder