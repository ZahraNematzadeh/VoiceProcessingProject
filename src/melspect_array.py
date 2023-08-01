import numpy as np
import tensorflow as tf

def melspect_array(melspect_data, var_leaf):
    if var_leaf: 
       leaf_list = np.array([item[0] for item in melspect_data])
       leaf_cell = tf.convert_to_tensor(leaf_list, dtype=tf.float32)
       #leaf_cell = tf.transpose(leaf_cell, [0, 2, 1])
       expanded_leaf_cell = tf.expand_dims(leaf_cell, axis=-1)
       final_array = tf.transpose(expanded_leaf_cell, perm=[0, 2, 1, 3])
    else :
        melspect_array = np.array([item[0] for item in melspect_data])
        melspect_array = np.stack(melspect_array, axis=0)
        melspect_array = melspect_array.reshape(
                                melspect_array.shape[0],
                                melspect_array.shape[1],
                                melspect_array.shape[2],
                                1)
        final_array = tf.convert_to_tensor(melspect_array, dtype=tf.float32)
    return final_array

