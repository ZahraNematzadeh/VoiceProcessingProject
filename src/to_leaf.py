import os
import tensorflow as tf
import leaf_audio.frontend as frontend

leaf = frontend.Leaf()

def to_leaf (audio_signal):
    leaf_list = []
    for sample in audio_signal:
        audio, filename, label = sample
        name, extension = os.path.splitext(filename)
        wave = tf.convert_to_tensor(audio)
        wave = tf.reshape(wave, (1, 16000))
        lf = leaf(wave)
        lf = tf.transpose(lf,[2,1,0])
        lf =lf[:, :, 0]
        leaf_list.append((lf.numpy(), filename, label))
    return leaf_list