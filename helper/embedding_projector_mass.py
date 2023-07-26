import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector
from sklearn.preprocessing import MinMaxScaler

def scale_minmax(melspec):
    scaler = MinMaxScaler()
    melspec = np.asarray(melspec, dtype=np.float32)
    minmax_melspec = scaler.fit_transform(melspec)
    return minmax_melspec

def pad_audio(signal, max_duration, sample_rate):
    padded_signal = np.array(signal)
    num_expected_samples = int(sample_rate * max_duration)
    if len(padded_signal) < num_expected_samples:
        num_missing_items = num_expected_samples - len(padded_signal)
        padded_signal = np.pad(padded_signal, (0, num_missing_items), mode="constant")
    elif len(padded_signal) > num_expected_samples:
        padded_signal = padded_signal[:num_expected_samples]
    elif len(padded_signal) == num_expected_samples:
        padded_signal = padded_signal
    return padded_signal

def audio_to_melspectrogram(audio_path, target_shape=(128, 218)):
    audio, sr = librosa.load(audio_path, sr=44100)
    audio = pad_audio(audio, 5, 44100)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    scaled_mel = scale_minmax(log_spectrogram)
    return scaled_mel

#%%
positive_folder = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/AllCleanedData/Positive'
data_csv = pd.read_csv('C:/Users/zahra/VoiceColab/dataset/metadata/data.csv', usecols=['MRN', 'Diagnosis','Diagnosis2', 'Type_of_Mass'])

signal_files = os.listdir(positive_folder)
signal_files_2 = [filename.split('-', 1)[1].replace('.wav', '') for filename in signal_files]

mrn_list = data_csv['MRN'].astype(str).tolist()

filtered_mrns = [mrn for mrn in mrn_list if str(mrn) in signal_files_2]
filtered_data = data_csv[data_csv['MRN'].isin(filtered_mrns)][['MRN','Type_of_Mass']]

metadata = [] 
feature_vectors = []
signal_mrn_type_of_mass_list = []

for filename in signal_files:
    mrn = filename.split('-', 1)[1].replace('.wav', '')
    if mrn in filtered_mrns:
        type_of_mass = filtered_data.loc[filtered_data['MRN'] == mrn, 'Type_of_Mass'].values[0]
        signal_mrn_type_of_mass_list.append((mrn, type_of_mass))
        
        metadata.append(type_of_mass)
        audio_path = os.path.join(positive_folder, filename)
        spectrogram = audio_to_melspectrogram(audio_path)
        feature_vectors.append(spectrogram)  
#%%
LOG_DIR = '/content/drive/My Drive/VoiceProcessingProject_Outputs/EmbeddingLogs/logsMass'
if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

metadata_file = os.path.join(LOG_DIR, 'metadata.tsv')
feature_file = os.path.join(LOG_DIR, 'feature_vectors.tsv')

feature_vectors = np.array(feature_vectors, dtype=np.float32)
metadata = np.array(metadata, dtype=str)

feature_vectors_2d = feature_vectors.reshape(feature_vectors.shape[0], -1)

np.savetxt(feature_file, feature_vectors_2d, delimiter='\t', fmt='%.8f')
#np.savetxt(metadata_file, metadata, delimiter='\t', fmt='%s', newline='\n')                   # Save the feature vectors and metadata

with open(metadata_file, 'w') as f:
    for label in metadata:
        f.write(f"{label}\n")

with tf.compat.v1.Session() as sess:                                            # Create a TensorFlow session
    features = tf.Variable(feature_vectors, name='features')                    # Create a variable to hold the feature vectors
    sess.run(tf.compat.v1.global_variables_initializer())                       # Initialize the variables
    saver = tf.compat.v1.train.Saver([features])                                # Create a saver to save the session
    checkpoint_path = os.path.join(LOG_DIR, 'feature_vectors.ckpt')             # Save the session checkpoint
    saver.save(sess, checkpoint_path)

emb_config = projector.ProjectorConfig()                                        # Configure the projector
embedding = emb_config.embeddings.add()                                         # Specify the embedding tensor
embedding.tensor_name = 'features'
embedding.metadata_path = metadata_file
embedding.tensor_path = feature_file

projector_path = os.path.join(LOG_DIR, 'projector_config.pbtxt')                # Save the projector configuration
with open(projector_path, 'w') as f:
    f.write(str(emb_config))

summary_writer = tf.summary.create_file_writer(LOG_DIR)
projector.visualize_embeddings(logdir=LOG_DIR, config=emb_config)               # Pass the log directory path as a string
summary_writer.close()