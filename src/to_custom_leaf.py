import os
from leaf_audio.postprocessing import PCENLayer
import tensorflow as tf
from leaf_audio import frontend, initializers
import functools

def to_custom_leaf(audio_signal):
   
    leaf_list = []
    
    n_filters = 71
    window_len = 25
    sample_rate = 16000
    preemp = True
    compression_fn = functools.partial(frontend.log_compression, log_offset=1e-5)
    complex_conv_init = initializers.GaborInit(sample_rate=sample_rate, min_freq=60., max_freq=5000.)
    learn_pooling = False 
    
    custom_leaf = frontend.Leaf(learn_pooling=learn_pooling,
                                n_filters=n_filters,
                                window_len=window_len,
                                sample_rate=sample_rate,
                                preemp=preemp,
                                compression_fn=compression_fn,
                                complex_conv_init=complex_conv_init)
   
    for sample in audio_signal:
        audio, filename, label = sample
        name, extension = os.path.splitext(filename)
        wave = tf.convert_to_tensor(audio)
        wave = tf.reshape(wave, (1, 16000))
        lf = custom_leaf(wave)  
        lf = tf.transpose(lf, [2, 1, 0])
        leaf_audio = lf[:, :, 0]
        leaf_list.append((leaf_audio.numpy(), filename, label))
    
    return leaf_list