import collections


def label_prediction(data_kfold):
    predicted_labels = []
    for i in range(len(data_kfold)):
        counts = collections.Counter(data_kfold.iloc[i])
        most_common_class = counts.most_common(1)[0][0]
        predicted_labels.append(most_common_class)
    return  predicted_labels