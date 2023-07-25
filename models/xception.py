import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (Dense, Dropout, Activation)


def xception(input_shape):

    base_model = Xception(weights='imagenet',
                      include_top=False,
                      input_shape=(input_shape))

    for layer in base_model.layers:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)                                                         # Add your own layers here
    x = Dense(512)(x)
    x = Activation('relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    for layer in base_model.layers:
        layer.trainable = True

    return model