import os
from df.enhance import enhance, init_df, load_audio, save_audio

model, df_state, _ = init_df()

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
audio_path = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/val/Negative/1-U2017561-23.wav'
model, df_state, _ = init_df()

audio, _ = load_audio(audio_path, sr=df_state.sr())
enhanced = enhance(model, df_state, audio)
save_audio("enhanced.wav", enhanced, df_state.sr())
'''