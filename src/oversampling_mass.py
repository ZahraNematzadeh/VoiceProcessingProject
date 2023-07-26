import os
from sklearn.utils import resample

def oversample_positive_class(data, folder_name):
    positive_samples = [sample for sample in data if sample[2] == 'Positive']
    negative_samples = [sample for sample in data if sample[2] == 'Negative']

    num_positive_samples_before = len(positive_samples)
    print("==================================================================================")
    print("Number of positive samples before balancing(" + folder_name + "):", num_positive_samples_before)
    num_negative_samples = len(negative_samples)
    print("Number of negative samples(" + folder_name + "):", num_negative_samples)

    num_to_add = num_negative_samples - num_positive_samples_before

    oversampled_positive_samples = resample(positive_samples, n_samples=num_to_add, replace=True,  random_state=0)
    balanced_data = negative_samples  + positive_samples

    num_positive_samples_after = len(oversampled_positive_samples)
    print("Number of added positive samples (" + folder_name + "):", num_positive_samples_after)
    print("==================================================================================")

    filename_count = {}
    for sample in oversampled_positive_samples:
        new_sample, filename, label = sample
        name, extension = os.path.splitext(filename)

        if name not in filename_count:
            filename_count[name] = 1
            new_filename = name + "_1" + extension
        else:
            filename_count[name] += 1
            new_filename = name + "_" + str(filename_count[name]) + extension

        balanced_data.append((new_sample, new_filename, label))

    #random.shuffle(balanced_data)
 
    return balanced_data
