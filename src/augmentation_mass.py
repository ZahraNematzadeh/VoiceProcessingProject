import os 
import random
import nlpaug.augmenter.audio as naa

sample_rate = 44100
def voice_augmentation(balanced_data):
    augmented_samples = []
    aug_tech = [ naa.LoudnessAug(zone = (0,1)),
                 #naa.CropAug(sampling_rate = sample_rate),
                 #naa.MaskAug(sampling_rate = sample_rate, zone=(0.2, 0.8),coverage= 0.7, mask_with_noise = False),
                 naa.NoiseAug(zone = (0,1),color='random'),
                 naa.PitchAug(zone = (0,1),sampling_rate = sample_rate)
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
