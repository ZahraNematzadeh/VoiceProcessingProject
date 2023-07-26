def get_dataset_name(folder_path):
    path_parts = folder_path_train.split('/')
    dataset_name = path_parts[-3]
    return dataset_name