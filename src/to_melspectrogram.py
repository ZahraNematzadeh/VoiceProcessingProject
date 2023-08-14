import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config.config import sample_rate

sample_rate = sample_rate
def scale_minmax(melspec):
    scaler = MinMaxScaler()
    melspec = np.asarray(melspec,dtype=np.float32)
    flattened_melspec = np.ravel(melspec)
    minmax_melspec = scaler.fit_transform(flattened_melspec.reshape(-1, 1))
    return minmax_melspec.reshape(melspec.shape)

def to_melspectrogram(input_data):
    melspect_data = []
    for augmented_sample in input_data:
        audio, filename, label = augmented_sample
        audio_np = np.asarray(audio)
        mel_spec = librosa.feature.melspectrogram(y=audio_np, sr=sample_rate, fmax=5000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        #mel_spec_db = scale_minmax(mel_spec_db)
        if mel_spec_db.shape[0] == 1:
            mel_spec_db = mel_spec_db[0]
        melspect_data.append((mel_spec_db, filename, label))

    return melspect_data


