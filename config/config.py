import os

K_fold = 10
Epoch = 100
Batch_size = 64

#---------------------------------- GDrive -------------------------------------
folder_path_train= '/content/drive/My Drive/Dataset/e/ClusteredData/big_mass_wav/train'
folder_path_test= '/content/drive/My Drive/Dataset/e/ClusteredData/big_mass_wav/val'

helper_output_path = '/content/drive/My Drive/VoicePathologyDetection_Outputs/HelpersOutputs'
final_output_path = '/content/drive/My Drive/VoicePathologyDetection_Outputs/FinalOutputs'
plots_output_path = '/content/drive/My Drive/VoicePathologyDetection_Outputs/Plots'

if not os.path.exists(helper_output_path):
        os.makedirs(helper_output_path)

if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

if not os.path.exists(plots_output_path):
        os.makedirs(plots_output_path)        


#folder_path_train= 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/train'
#folder_path_test= 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/val'

#folder_path_train_leaf = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/train'
#folder_path_test_leaf = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav/train'

#helper_output_path = 'C:/Users/zahra/VoiceColab/outputs/HelpersOutputs'
#final_output_path = 'C:/Users/zahra/VoiceColab/outputs/FinalOutputs'
#plots_output_path = 'C:/Users/zahra/VoiceColab/outputs/Plots'