from src.padding_audio import process_folder
from src.oversampling_mass import oversample_positive_class
from src.augmentation_mass import voice_augmentation
from src.converting_to_melspectrogram import audio_to_melspect
from src.label_encoder import label_encoder
from src.kfold_training import kfold_training
from src.plot_each_fold import plot_each_fold
from src.plot_avg_fold import plot_avg_fold
from src.label_prediction import label_prediction
from src.confusion_mat import confusion_mat
from src.classification_report import classification_reports
from src.roc_curve_function import roc_curve_function
from src.all_roc_curves import all_roc_curves
from src.melspect_array import melspect_array
from src.get_dataset_name import get_dataset_name
from models.cnn import cnn_function
from models.inceptionv3 import inceptionv3
from models.resnet50 import resnet50
from models.xception import xception
import os
import pickle
import numpy as np
from keras import callbacks
import matplotlib.pyplot as plt


#%%
folder_path_train= '/content/drive/My Drive/Dataset/e/ClusteredData/big_mass/train'
folder_path_test= '/content/drive/My Drive/Dataset/e/ClusteredData/big_mass/val'
padded_train = []
padded_test = []

dataset_name = get_dataset_name(folder_path_train)

process_folder(folder_path_train, padded_train)
process_folder(folder_path_test, padded_test)

#%%
balanced_train_data = oversample_positive_class(padded_train, dataset_name="Train")
balanced_test_data = oversample_positive_class(padded_test,  dataset_name="Test")

if not os.path.exists('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs'):
        os.makedirs('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs')

output_file_path_train = os.path.join('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs', "balanced_train_data.pkl")
with open(output_file_path_train, "wb") as file:
       pickle.dump(balanced_train_data, file)

output_file_path_test = os.path.join('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs', "balanced_test_data.pkl")       
with open(output_file_path_test, "wb") as file:
       pickle.dump(balanced_test_data, file)

#%%
augmented_train = voice_augmentation(balanced_train_data)
augmented_test = voice_augmentation(balanced_test_data)
print('================ Audios have been augmented successfully ===============')

output_file_path_train = os.path.join('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs', "augmented_train.pkl")
with open(output_file_path_train, "wb") as file:
       pickle.dump(augmented_train, file)

output_file_path_test = os.path.join('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs', "augmented_test.pkl")       
with open(output_file_path_test, "wb") as file:
       pickle.dump(augmented_test, file)
#%%       
from helper.counting_augmented_samples import counting_aug_samples      
positive_count,negative_count,total_count  = counting_aug_samples(augmented_train)
print("Number of Positive samples in train:", positive_count)
print("Number of Negative samples in train:", negative_count)
print('Total Number of samples in train:', total_count)

positive_count,negative_count,total_count  = counting_aug_samples(augmented_test)
print("Number of Positive samples in test:", positive_count)
print("Number of Negative samples in test:", negative_count)
print('Total Number of samples in test:', total_count)
#%%
melspect_train_data = audio_to_melspect(augmented_train)
melspect_test_data = audio_to_melspect(augmented_test)
print('============== Audios have been converted to melspectrogram successfully ===============')

output_file_path_train = os.path.join('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs', "melspect_train_data.pkl")
with open(output_file_path_train, "wb") as file:
       pickle.dump(melspect_train_data, file)

output_file_path_test = os.path.join('/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs', "melspect_test_data.pkl")       
with open(output_file_path_test, "wb") as file:
       pickle.dump(melspect_test_data, file)

#%%
y_train_one_hot,_ = label_encoder(melspect_train_data)
y_test_one_hot, y_test_encoded = label_encoder(melspect_test_data)

#%%
color_space = input("Enter 'g' for grayscale or 'r' for RGB: ")
if color_space.lower() == 'g':
    melspect_train_array = melspect_array(melspect_train_data)
    melspect_test_array = melspect_array(melspect_test_data)
    input_shape = melspect_train_array.shape[1:]     #input_shape = (128,431,1)
    num_classes = 2
    model = cnn_function(input_shape, 2)
    model.summary()
elif color_space.lower() == 'r':
    melspect_train_array = melspect_array(melspect_train_data)
    melspect_test_array = melspect_array(melspect_test_data)
    melspect_train_array = np.repeat(melspect_train_array, 3, axis=-1)
    melspect_test_array = np.repeat(melspect_test_array, 3, axis=-1)
    input_shape = melspect_train_array.shape[1:]
    transfer_learning = input("Enter 'r' for Resnet50 or 'i' for InceptionV3 or 'x' for Xception: ")
    if transfer_learning.lower() == 'r':
        model = resnet50(input_shape)
        model.summary()
    elif transfer_learning.lower() == 'i':
        model = inceptionv3(input_shape)
        model.summary()
    elif transfer_learning.lower() == 'x':
        model = xception(input_shape)
        model.summary()
    else:
        print("Invalid transfer learning option. Please enter 'r' for Resnet50, 'i' for InceptionV3, or 'x' for Xception.")
else:
    print("Invalid color space option. Please enter 'g' for grayscale or 'r' for RGB.")
         
#%%
#earlystopping = callbacks.EarlyStopping(monitor = "val_loss", mode= "min",
                                        #patience= 10, restore_best_weights= True)

scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                        patience=5, min_lr=1e-8, verbose=1)

checkpointer = callbacks.ModelCheckpoint(filepath='/content/drive/My Drive/VoiceProcessingProject_Outputs/FinalOutputs/e_cnn_bigMass_augtest.hdf5',
                                         verbose=1, save_best_only=True)
#%% Training
_,_, test_count = counting_aug_samples(augmented_test)
data_kfold, model_history = kfold_training(melspect_train_array ,melspect_test_array,
                                           y_train_one_hot, y_test_one_hot,
                                           test_count, model, scheduler,
                                           checkpointer, k_fold=3, num_epochs=3)
#%% saving models
with open('/content/drive/My Drive/VoiceProcessingProject_Outputs/FinalOutputs/modelhistory_cnn_e_big_testaug.pkl', 'wb') as f:                                      # Save model_history
    pickle.dump(model_history, f)
    
with open('/content/drive/My Drive/VoiceProcessingProject_Outputs/FinalOutputs/data_kfold_cnn_e_big_testaug.pkl', 'wb') as f:
    pickle.dump(data_kfold, f)                                                               # Save data_kfold for prediction

# Loading models
with open('/content/drive/My Drive/VoiceProcessingProject_Outputs/FinalOutputs/modelhistory_cnn_e_big_testaug.pkl', 'rb') as f:                                      # Load model_history
    loaded_model_history = pickle.load(f)
    
with open('/content/drive/My Drive/VoiceProcessingProject_Outputs/FinalOutputs/data_kfold_cnn_e_big_testaug.pkl', 'rb') as f:                                        # Load data_kfold for prediction
    data_kfold = pickle.load(f)
#%% Plotting learning curves
plot_each_fold(model_history, dataset_name)
plot_avg_fold(model_history, dataset_name)

#%%
predicted_labels = label_prediction(data_kfold)
true_labels = y_test_encoded

np.save('/content/drive/My Drive/VoiceProcessingProject_Outputs/FinalOutputs/predicted_labels_e_cnn_big_testaug.npy', np.array(predicted_labels))
np.save('/content/drive/My Drive/VoiceProcessingProject_Outputs/FinalOutputs/true_labels_bigmass_e_cnn_big_testaug.npy', np.array(true_labels))


#%%
confusion_mat(true_labels, predicted_labels, dataset_name)
classification_reports(true_labels, predicted_labels)
roc_curve_function(true_labels, predicted_labels, dataset_name)

#%%
'''
true_labels_original = np.load('true_labels_original_5_e_cnn.npy')
predicted_labels_original = np.load('predicted_labels_original_5_e_cnn.npy')
    
true_labels_bigmass = np.load('true_labels_bigmass_5_e_cnn.npy')
predicted_labels_bigmass = np.load('predicted_labels_bigmass_5_e_cnn.npy')
    
true_labels_smallmass = np.load('true_labels_smallmass_5_e_cnn.npy')
predicted_labels_smallmass = np.load('predicted_labels_smallmass_5_e_cnn.npy')
    
true_labels_mediummass = np.load('true_labels_Mediummass_5_e_cnn.npy')
predicted_labels_mediummass = np.load('predicted_labels_Mediummass_5_e_cnn.npy')

all_roc_curves(true_labels_original,predicted_labels_original,
               true_labels_bigmass,predicted_labels_bigmass,
               true_labels_smallmass,predicted_labels_smallmass,
               true_labels_mediummass,predicted_labels_mediummass)
'''