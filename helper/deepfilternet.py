import os
from df.enhance import enhance, init_df, load_audio, save_audio

model, df_state, _ = init_df()

def deepfilternet(audio_path, output_dir, set_name):
    
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
                
                audio, _ = load_audio(file_path, sr=df_state.sr())
                enhanced = enhance(model, df_state, audio)
                
                output_file_path = os.path.join(output_label_dir, filename)
                save_audio(output_file_path, enhanced, df_state.sr())


if __name__ == '__main__':
    folder_path_train = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/train'
    folder_path_test = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/val'
    
    output_dir = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav_deepfilternet'
    
    deepfilternet(folder_path_train, output_dir, 'train') 
    deepfilternet(folder_path_test, output_dir, 'val') 


'''
def deepfilternet(audio_path):
    
    enhanced_audio = []
    for root, dirs, files in os.walk(audio_path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                label = os.path.basename(root)
                audio, _ = load_audio(file_path, sr=df_state.sr())
                enhanced = enhance(model, df_state, audio)
                enhanced_audio.append((enhanced, filename, label))
    return enhanced_audio, df_state.sr()
'''
