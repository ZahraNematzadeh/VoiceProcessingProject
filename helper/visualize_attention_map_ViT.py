import numpy as np
from vit_keras import visualize

output_train = np.load('C:/Users/zahra/VoiceColab/outputs/HelpersOutputs/1_e/big_mass/Melspectrogram/ViT/output_train.npy')

image = output_train[0][0]

attention_map = visualize.attention_map(model = model, image = image)
attention_map = visualize.attention_map(model=model.layers[0], image=image)


attention_map1 = visualize.attention_map(model = vit_model_t.get_layer('vit_model'), image = res)

fig = plt.figure(figsize=(20,20))
ax = plt.subplot(1, 2, 1)
ax.axis('off')
ax.set_title('Original')
_ = ax.imshow(res)

ax = plt.subplot(1, 2, 2)
ax.axis('off')
ax.set_title('Attention Map')
_ = ax.imshow(attention_map1)