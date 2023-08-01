import numpy as np



padded_train = []
padded_test = []
padded_list = []

def pad_audio_leaf(signal, max_duration, sample_rate):
    
    for item in signal:
        signal_to_pad, filename, label = item[0], item[1], item[2]
      
        num_expected_samples = int(sample_rate * max_duration)
        if len(signal_to_pad) < num_expected_samples:
            num_missing_items = num_expected_samples - len(signal_to_pad)
            padded_signal = np.pad(signal_to_pad, (0, num_missing_items), mode="constant")
        elif len(padded_signal) > num_expected_samples:
            padded_signal = padded_signal[:num_expected_samples]
        elif len(padded_signal) == num_expected_samples:
            padded_signal = padded_signal
            
        padded_list.append((padded_signal, filename, label))
            
            
    return padded_signal