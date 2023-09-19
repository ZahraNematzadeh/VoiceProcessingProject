import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
import matplotlib.pyplot as plt

output_train = np.load('C:/Users/zahra/VoiceColab/outputs/HelpersOutputs/1_e/big_mass/Melspectrogram/ViT/output_train.npy')

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

def visualize_patches(image, image_size, patch_size):
    
    plt.figure(figsize=(4, 4))
    #image = output_train[0]
    image = image
    plt.imshow(image)
    plt.title(f"Original", fontsize=16)
    plt.axis('off')

    resized_image = tf.image.resize([image], size = (image_size, image_size))
    patches = Patches(patch_size)(resized_image)
    print(f'Image size: {image_size} X {image_size}')
    print(f'Patch size: {patch_size} X {patch_size}')
    print(f'Patches per image: {patches.shape[1]}')
    print(f'Elements per patch: {patches.shape[-1]}')
    
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img)
        plt.axis('off')
    plt.suptitle("Patch Set", fontsize=16)
    
    
if __name__ == '__main__':
    for sample_index in range(5):
        image = output_train[sample_index]
        visualize_patches(image, 224, 32)
