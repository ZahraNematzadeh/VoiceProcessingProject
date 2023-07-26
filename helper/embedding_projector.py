import os
import librosa
import numpy as np
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

LOG_DIR = '/content/drive/My Drive/VoiceProcessingProject_Outputs/EmbeddingLogs/logs'
if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

metadata_file = os.path.join(LOG_DIR, 'metadata.tsv')
feature_file = os.path.join(LOG_DIR, 'feature_vectors.tsv')

positive_folder = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/AllCleanedData/Positive'
#negative_folder = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/CleanedData/train/Negative'

feature_vectors = []
metadata = []

for filename in os.listdir(positive_folder):                                    # Convert positive audio to mel spectrograms and save the features
    audio_path = os.path.join(positive_folder, filename)
    spectrogram = audio_to_melspectrogram(audio_path)
    feature_vectors.append(spectrogram)
    metadata.append(['positive'])
'''
for filename in os.listdir(negative_folder):
    audio_path = os.path.join(negative_folder, filename)
    spectrogram = audio_to_melspectrogram(audio_path)
    feature_vectors.append(spectrogram)
    metadata.append(['negative'])
'''
feature_vectors = np.array(feature_vectors, dtype=np.float32)
metadata = np.array(metadata, dtype=str)


feature_vectors_2d = feature_vectors.reshape(feature_vectors.shape[0], -1)

np.savetxt(feature_file, feature_vectors_2d, delimiter='\t', fmt='%.8f')
np.savetxt(metadata_file, metadata, delimiter='\t', fmt='%s', newline='\n')                   # Save the feature vectors and metadata

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