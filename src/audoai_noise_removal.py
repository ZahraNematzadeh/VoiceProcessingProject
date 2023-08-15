import os
from audoai.noise_removal import NoiseRemovalClient


def audoai_noise_removal(audio_path):
    cleaned_audio = []
    noise_removal = NoiseRemovalClient(api_key="fc980a1f752e615ff557f609bc6c8e0f")
    for root, dirs, files in os.walk(audio_path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                label = os.path.basename(root)
                result = noise_removal.process(file_path)
                cleaned_audio.appened((result, filename, label))
    return cleaned_audio
    
    
    
    