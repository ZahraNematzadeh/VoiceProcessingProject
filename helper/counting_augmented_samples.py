import pickle


with open('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs/augmented_train.pkl', "rb") as file:
    augmented_train = pickle.load(file)
    
with open('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs/augmented_test.pkl', "rb") as file:
    augmented_test = pickle.load(file)
    

def counting_aug_samples(augmented_data):
    DATA = augmented_data
    positive_count = 0
    negative_count = 0
    for item in DATA:
        label = item[2]
        if label == "Positive":
            positive_count += 1
        elif label == "Negative":
            negative_count += 1
    test_count = positive_count + negative_count
    return positive_count, negative_count, test_count

positive_tr_count, negative_tr_count, test_tr_count = counting_aug_samples(augmented_train)
positive_ts_count, negative_ts_count, test_ts_count = counting_aug_samples(augmented_test)


