import pickle

def counting_aug_samples(augmented_data, dataset_folder_helper):

    with open(dataset_folder_helper + augmented_data, "rb") as file:
        augmented_data = pickle.load(file)

    positive_count = 0
    negative_count = 0
    for item in augmented_data:
        label = item[2]
        if label == "Positive":
            positive_count += 1
        elif label == "Negative":
            negative_count += 1
    test_count = positive_count + negative_count
    return positive_count, negative_count, test_count



