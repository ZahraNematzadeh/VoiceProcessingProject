import os
from audoai.noise_removal import NoiseRemovalClient


def audoai_noise_removal(audio_path, output_dir, set_name):
    
    noise_removal = NoiseRemovalClient(api_key="fc980a1f752e615ff557f609bc6c8e0f")
    for root, dirs, files in os.walk(audio_path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                label = os.path.basename(root)
                result = noise_removal.process(file_path)
                output_set_dir = os.path.join(output_dir, set_name)
                output_label_dir = os.path.join(output_set_dir, label)

                if not os.path.exists(output_set_dir):
                    os.makedirs(output_set_dir, exist_ok=True)
                
                if not os.path.exists(output_label_dir):
                   os.makedirs(output_label_dir, exist_ok=True)
                
                output_file_path = os.path.join(output_label_dir, filename)
                result.save(output_file_path)
    
    
if __name__ == '__main__':
    folder_path_train = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/train'
    folder_path_test = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/val'
    
    output_dir = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav_audoai'
    
    audoai_noise_removal(folder_path_train, output_dir, 'train') 
    audoai_noise_removal(folder_path_test, output_dir, 'val') 


    
    