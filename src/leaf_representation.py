import numpy as np
import tensorflow as tf
import leaf_audio.frontend as frontend


def leaf_representation(data_list):
    leaf_list = []
    leaf = frontend.Leaf()  
    for data_entry in data_list:
        audio_array, filename, label = data_entry
        audio_tensor = tf.convert_to_tensor(audio_array, dtype=tf.float32)[tf.newaxis, :]
        
        lf = leaf(audio_tensor)  
        lf = tf.transpose(lf, [2, 1, 0])
        leaf_audio = lf[:, :, 0]
        leaf_list.append((leaf_audio.numpy(), filename, label))
    
    return leaf_list