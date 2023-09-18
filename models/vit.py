#!pip install vit_keras

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa

from vit_keras import vit


def vision_transformer(image_size,num_classes, patch_size):
        vit_model = vit.vit_b16(
                image_size = image_size,
                activation = 'softmax',
                pretrained = True,
                include_top = False,
                pretrained_top = False,
                classes = num_classes)

        class Patches(L.Layer):
            def __init__(self, patch_size):
                super(Patches, self).__init__()
                self.patch_size = patch_size
        
            def call(self, images):
                batch_size = tf.shape(images)[0]
                patches = tf.image.extract_patches(
                    images = images,
                    sizes = [1, self.patch_size, self.patch_size, 1],
                    strides = [1, self.patch_size, self.patch_size, 1],
                    rates = [1, 1, 1, 1],
                    padding = 'VALID',
                )
                patch_dims = patches.shape[-1]
                patches = tf.reshape(patches, [batch_size, -1, patch_dims])
                return patches
    
        model = tf.keras.Sequential([
                vit_model,
                tf.keras.layers.Flatten(),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation = tfa.activations.gelu),
                tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
                tf.keras.layers.Dense(2, 'softmax')
            ],
            name = 'vision_transformer')

        return model