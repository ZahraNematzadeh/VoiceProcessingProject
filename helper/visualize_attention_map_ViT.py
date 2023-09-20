import numpy as np
import matplotlib.pyplot as plt
from vit_keras import visualize
from tensorflow.keras.models import load_model

loaded_model = load_model('C:/Users/zahra/VoiceColab/outputs/FinalOutputs/dataset/big_mass/Melspectrogram/ViT/dataset_model_ViT.hdf5')
output_train = np.load('C:/Users/zahra/VoiceColab/outputs/HelpersOutputs/dataset/big_mass/Melspectrogram/ViT/output_train.npy')

image = output_train[0]

#attention_map = visualize.attention_map(model=loaded_model.layers[0], image=image)
attention_map = visualize.attention_map(model = loaded_model.get_layer('vit-b16'), image = image)

fig = plt.figure(figsize=(20,20))
ax = plt.subplot(1, 2, 1)
ax.axis('off')
ax.set_title('Original', fontsize=32)
_ = ax.imshow(image)

ax = plt.subplot(1, 2, 2)
ax.axis('off')
ax.set_title('Attention Map', fontsize=32)
_ = ax.imshow(attention_map)

plt.show()

'''
import librosa.display
spec = np.mean(image, axis=-1)
spec = np.squeeze(spec)
librosa.display.specshow(spec)
'''