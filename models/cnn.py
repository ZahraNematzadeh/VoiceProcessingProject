import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
                                  

def cnn_function(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Conv2D(32,(3,3),  input_shape= input_shape, kernel_regularizer=regularizers.l2(0.0005)),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPool2D((2,2), padding="same"),

        keras.layers.Conv2D(64,(3,3), kernel_regularizer=regularizers.l2(0.0005)),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPool2D((2,2), padding="same"),

        keras.layers.Conv2D(128,(3,3), kernel_regularizer=regularizers.l2(0.0005)),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPool2D((2,2), padding="same"),

        keras.layers.Conv2D(256,(3,3), kernel_regularizer=regularizers.l2(0.0005)),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPool2D((2,2), padding="same"),

        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256),
        keras.layers.Activation('relu'),
        #keras.layers.Dense(128, activation='relu'),
        #keras.layers.Dense(64, activation='relu'),
        #keras.layers.Dropout(0.7),
        keras.layers.Dense(num_classes, activation='softmax')])

    return model


    