import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def label_encoder(melspect_data):
    y = np.array([item[2] for item in melspect_data])
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_one_hot = to_categorical(y_encoded)
    return y_one_hot, y_encoded
