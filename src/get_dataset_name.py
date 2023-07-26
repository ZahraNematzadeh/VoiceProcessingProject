def get_dataset_name(folder_path):
    path_parts = folder_path.split('/')
    dataset_name = path_parts[-4]
    return dataset_name