import gc
import librosa
import numpy as np

#%%

def pad_audio(audio, sr, max_duration):
    padded_list = []
    for data in audio:
      signal = data[0]
      filename = data[1]
      label = data[2]
      
      signal_np = signal.numpy()
      signal_np = librosa.resample(signal_np,  orig_sr= 44100, target_sr=48000)
      
      len_signal_np = signal_np.shape[1]
      num_expected_samples = int(sr * max_duration)
      
      print('num_expected_samples',num_expected_samples)
      
      if len_signal_np < num_expected_samples:
          num_missing_items = (num_expected_samples - len_signal_np)
          print('num_missing_items', num_missing_items)
          print('len_signal_np', len_signal_np)
          
          padded_signal = np.pad(signal_np, (0, num_missing_items), mode="constant")
          
          print('====================')
      elif len_signal_np > num_expected_samples:
           padded_signal = signal_np[:num_expected_samples]
           print('222')
      else:
          padded_signal = signal_np
          print('333')
      
      padded_list.append((padded_signal, filename, label))
      gc.collect()
    
    return padded_list

