import os 
from scipy.io import wavfile
import noisereduce as nr


def spectral_gating(audio_path, output_dir, set_name):
    
    for root, dirs, files in os.walk(audio_path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                label = os.path.basename(root)
                
                output_set_dir = os.path.join(output_dir, set_name)
                output_label_dir = os.path.join(output_set_dir, label)

                if not os.path.exists(output_set_dir):
                    os.makedirs(output_set_dir, exist_ok=True)
                
                if not os.path.exists(output_label_dir):
                   os.makedirs(output_label_dir, exist_ok=True)
                
                rate, data = wavfile.read(file_path)
                reduced_noise = nr.reduce_noise(y=data, sr=rate)
                
                output_file_path = os.path.join(output_label_dir, filename)
                wavfile.write(output_file_path, rate, reduced_noise)    
    
if __name__ == '__main__':
    folder_path_train = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/train'
    folder_path_test = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/val'
    
    output_dir = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav_spectral_gating'
    
    spectral_gating(folder_path_train, output_dir, 'train') 
    spectral_gating(folder_path_test, output_dir, 'val') 

