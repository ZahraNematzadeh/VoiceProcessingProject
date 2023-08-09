def get_avg_amp(wave_data, sr, var_leaf):
    final_wave = []
    for data in wave_data:
        signal = data[0]
        filename = data[1]
        label = data[2]
          
        if var_leaf==True:
            signal = signal[0]
        average_amplitude = sum(abs(signal)) / len(signal)
        modified_audio = signal - average_amplitude
        final_wave.append((modified_audio, filename, label))
        
    return final_wave
