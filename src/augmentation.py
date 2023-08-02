import os 
import random
import nlpaug.augmenter.audio as naa
from audiomentations import AddGaussianNoise


def augmentation(balanced_data, var_leaf):
    if var_leaf:
        sample_rate = 16000
    else:
        sample_rate = 44100
        
    augmented_samples = []
    aug_tech = [ naa.LoudnessAug(zone = (0,1)),
                 #naa.CropAug(sampling_rate = sample_rate),
                 #naa.MaskAug(sampling_rate = sample_rate, zone=(0.2, 0.8),coverage= 0.7, mask_with_noise = False),
                 naa.NoiseAug(zone = (0,1), color='random'),
                 naa.PitchAug(sampling_rate = sample_rate, zone = (0,1))
                ] 

    for sample in balanced_data:
        audio, filename, label = sample
        name, extension = os.path.splitext(filename)
        if '_' in name:
            aug_idx = random.randint(0, len(aug_tech)- 1)
            aug_technique = aug_tech[aug_idx]
            if var_leaf:
                if isinstance(aug_technique, naa.NoiseAug):
                    transform = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015,p=1.0)
                    augmented_audio = transform(audio, sample_rate = 16000)
                    augmented_samples.append((augmented_audio, filename, label))

                else:
                    augmented_audio = aug_technique.augment(audio)
                    augmented_samples.append((augmented_audio, filename, label))
            else:
                augmented_audio = aug_technique.augment(audio)
                augmented_samples.append((augmented_audio, filename, label))
        else:
            augmented_samples.append(sample)
    return augmented_samples
'''
def voice_augmentation(balanced_data, var_leaf):
    if var_leaf:
        sample_rate = 16000
    else:
        sample_rate = 44100
        
    augmented_samples = []
    aug_tech = [ naa.LoudnessAug(zone = (0,1)),
                 #naa.CropAug(sampling_rate = sample_rate),
                 #naa.MaskAug(sampling_rate = sample_rate, zone=(0.2, 0.8),coverage= 0.7, mask_with_noise = False),
                 naa.NoiseAug(zone = (0,1), color='random'),
                 naa.PitchAug(sampling_rate = sample_rate, zone = (0,1))
                ] 

    for sample in balanced_data:
        audio, filename, label = sample
        name, extension = os.path.splitext(filename)
        if '_' in name:
            aug_idx = random.randint(0, len(aug_tech)- 1)
            aug_technique = aug_tech[aug_idx]
            augmented_audio = aug_technique.augment(audio)
            augmented_samples.append((augmented_audio, filename, label))
           
        else:
            augmented_samples.append(sample)
    return augmented_samples
'''
