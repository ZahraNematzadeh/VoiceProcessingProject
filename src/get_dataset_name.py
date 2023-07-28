def get_dataset_name(folder_path, num):
    path_parts = folder_path.split('/')
    dataset_name = path_parts[num]
    return dataset_name