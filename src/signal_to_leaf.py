import os
import tensorflow as tf
import leaf_audio.frontend as frontend

leaf = frontend.Leaf()
def signal_to_leaf(folder_path):
    
    leaf_list = []
    labels = ['Positive', 'Negative']
    for label in labels:
        subfolder_path = os.path.join(folder_path, label)
        for audio_file in os.listdir(subfolder_path):
            if audio_file.endswith('.wav'):
                file_path = os.path.join(subfolder_path, audio_file)
                raw_audio = tf.io.read_file(file_path)
                waveform = tf.audio.decode_wav(raw_audio, desired_channels=1, desired_samples=16000)
                waveform = tf.transpose(waveform.audio)
                lf = leaf(waveform)
                lf = tf.transpose(lf,[2,1,0])
                leaf_audio =lf[:, :, 0]
                filename = os.path.splitext(audio_file)[0]
                leaf_list.append((leaf_audio.numpy(), filename, label))
    return leaf_list

    
    
    
    
   
   