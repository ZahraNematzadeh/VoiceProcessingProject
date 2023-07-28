import os
import tensorflow as tf
from leaf_audio import frontend, initializers
import functools

def custom_leaf_representation(folder_path):
    leaf_list = []
    
    n_filters = 40
    window_len = 50
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

    labels = ['Positive', 'Negative']
    for label in labels:
        subfolder_path = os.path.join(folder_path, label)
        for audio_file in os.listdir(subfolder_path):
            if audio_file.endswith('.wav'):
                file_path = os.path.join(subfolder_path, audio_file)
                raw_audio = tf.io.read_file(file_path)
                waveform = tf.audio.decode_wav(raw_audio, desired_channels=1, desired_samples=16000)
                waveform = tf.transpose(waveform.audio)
                
                lf = custom_leaf(waveform)  
                lf = tf.transpose(lf, [2, 1, 0])
                leaf_audio = lf[:, :, 0]
                
                filename = os.path.splitext(audio_file)[0]
                leaf_list.append((leaf_audio.numpy(), filename, label))
    
    return leaf_list