import splitfolders

input_folder= 'C:/Users/zahra/VoiceColab/dataset/1_e/ConcatenatedAudio/ClusteredData/medium_mass'
output= 'C:/Users/zahra/VoiceColab/dataset/1_e/ConcatenatedAudio/test_train/ClusteredData/medium_mass'
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .2))