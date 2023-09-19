from src.pad_audio import pad_audio
from src.learning_selection_function import learning_selection_function
from src.oversampling_mass import oversample_positive_class
from src.augmentation import augmentation
from src.to_melspectrogram import to_melspectrogram
#from src.decode_wave import decode_wave
#from src.to_leaf import to_leaf
#from src.to_custom_leaf import to_custom_leaf
from src.input_array import input_array
from src.label_encoder import label_encoder
from src.kfold_training import kfold_training
from src.plot_each_fold import plot_each_fold
from src.plot_avg_fold import plot_avg_fold
from src.label_prediction import label_prediction
from src.confusion_mat import confusion_mat
from src.classification_report import classification_reports
from src.roc_curve_function import roc_curve_function
#from src.all_roc_curves import all_roc_curves
#from src.leaf_representation import leaf_representation
from src.get_informative_chunk import get_informative_chunk
from src.get_avg_amp import get_avg_amp
from src.noise_profiling import noise_preparing, find_matching_noise, noise_removal


from src.mono_to_color import mono_to_color
from src.vit_resize_image import ViT_resize_image

from models.cnn import cnn_function
from models.inceptionv3 import inceptionv3
from models.resnet50 import resnet50
from models.xception import xception
from models.vit import vision_transformer

from config.config import (folder_path_train, folder_path_test, K_fold, 
                            Epoch, Batch_size, sample_rate, max_duration, num_classes)

import os
import pickle
import numpy as np
from keras import callbacks

#-------------------------------------------------------------------------------
KFOLD = K_fold
EPOCH = Epoch
BATCH = Batch_size
sr = sample_rate
max_duration = max_duration
path_train = folder_path_train
path_test = folder_path_test
num_classes = num_classes

#------------------------------------------------------------------------------
'''
visualizing_selection = input("Enter 's' to reduce noise, or 'w' to work on whole audio:")
if visualizing_selection.lower() == 's':
    var_chunk = True
    
elif visualizing_selection.lower() == 'w':
    var_chunk = False  
 '''  
#------------------------------------------------------------------------------
visualizing_selection = input("Enter 'm' to convert audios to Melspectrogram --or-- 'l' to convert them to Leaf:")
if visualizing_selection.lower() == 'm':
    var_leaf = False
    dataset_folder_helper, dataset_folder_final, dataset_folder_plots, dataset_name, var_cnn,var_resnet, var_inception, var_xception, var_vit = learning_selection_function(path_train, path_test, var_leaf)
    wave_train = pad_audio(path_train, sample_rate, max_duration = max_duration )
    wave_test = pad_audio(path_test, sample_rate, max_duration = max_duration)

elif visualizing_selection.lower() == 'l':
    var_leaf = True
    dataset_folder_helper, dataset_folder_final, dataset_folder_plots, dataset_name, var_cnn,var_resnet, var_inception, var_xception, var_vit = learning_selection_function(path_train, path_test, var_leaf)
    wave_train = decode_wave(path_train)
    wave_test = decode_wave(path_test)  
#------------------------------------------------------------------------------
train_data = wave_train
test_data = wave_test 
print(sr)
#------------------------------------------------------------------------------
balanced_train_data = oversample_positive_class(train_data, folder_name="Train")
balanced_test_data = oversample_positive_class(test_data, folder_name="Test")

output_file_path_train = os.path.join(dataset_folder_helper, "balanced_train_data.pkl")
with open(output_file_path_train, "wb") as file:
       pickle.dump(balanced_train_data, file)

output_file_path_test = os.path.join(dataset_folder_helper, "balanced_test_data.pkl")       
with open(output_file_path_test, "wb") as file:
       pickle.dump(balanced_test_data, file)
#------------------------------------------------------------------------------
'''
if var_chunk:
    #train_chunk  = get_informative_chunk(balanced_train_data, sr, var_leaf)
    #train_chunk  = get_avg_amp(balanced_train_data, sr, var_leaf)
    noise_clips = noise_preparing(path_train, var_train=True, var_test=False)
    train_chunk = noise_removal(balanced_train_data, noise_clips)
    
    final_train = train_chunk
    print('====== Noise have been removed from TRAINING set successfully =========')
    
    #test_chunk = get_informative_chunk(balanced_test_data, sr, var_leaf)
    #test_chunk  = get_avg_amp(balanced_test_data, sr, var_leaf)
    noise_clips = noise_preparing(path_test, var_train= False, var_test=True)
    test_chunk = noise_removal(balanced_test_data, noise_clips) 
    
    final_test = test_chunk
    print('====== Noise have been removed from TEST set successfully ========') 
else:
    final_train = balanced_train_data
    final_test = balanced_test_data
'''
#------------------------------------------------------------------------------
augmented_train = augmentation(balanced_train_data, var_leaf)
augmented_test = augmentation(balanced_test_data, var_leaf)
print('==================== Audios have been augmented successfully =====================')

output_file_path_train = os.path.join(dataset_folder_helper, "augmented_train.pkl")
with open(output_file_path_train, "wb") as file:
       pickle.dump(augmented_train, file)

output_file_path_test = os.path.join(dataset_folder_helper, "augmented_test.pkl")       
with open(output_file_path_test, "wb") as file:
       pickle.dump(augmented_test, file)
#-------------------------------------------------------------------------------      
from helper.count_augmented_samples import counting_aug_samples

positive_count,negative_count,total_count  = counting_aug_samples(dataset_folder_helper,'augmented_train.pkl')
print("Number of Positive samples in train:", positive_count)
print("Number of Negative samples in train:", negative_count)
print('Total Number of samples in train:', total_count)

positive_count,negative_count,test_count  = counting_aug_samples(dataset_folder_helper,'augmented_test.pkl')
print("Number of Positive samples in test:", positive_count)
print("Number of Negative samples in test:", negative_count)
print('Total Number of samples in test:', test_count)
#-------------------------------------------------------------------------------
if var_leaf == False:
    train_data = to_melspectrogram(augmented_train)
    test_data = to_melspectrogram(augmented_test)
    
    print('============= Audios have been converted to Melspectrogram successfully ==========')

    output_file_path_train = os.path.join(dataset_folder_helper, "melspect_train_data.pkl")
    with open(output_file_path_train, "wb") as file:
           pickle.dump(train_data, file)

    output_file_path_test = os.path.join(dataset_folder_helper, "melspect_test_data.pkl")       
    with open(output_file_path_test, "wb") as file:
           pickle.dump(test_data, file)

else:   
    #train_data  = to_leaf(augmented_train)
    #test_data = to_leaf(augmented_test)
    
    train_data  = to_custom_leaf(augmented_train)
    test_data = to_custom_leaf(augmented_test)
    
    output_file_path_train = os.path.join(dataset_folder_helper, "leaf_train_data.pkl")
    with open(output_file_path_train, "wb") as file:
           pickle.dump(train_data, file)

    output_file_path_test = os.path.join(dataset_folder_helper, "leaf_test_data.pkl")       
    with open(output_file_path_test, "wb") as file:
           pickle.dump(test_data, file)
    
    print('============= Audios have been converted to Leaf successfully ==========')

#-------------------------------------------------------------------------------
y_train_one_hot,_ = label_encoder(train_data)
y_test_one_hot, y_test_encoded = label_encoder(test_data)

#-------------------------------------------------------------------------------
if var_cnn:
    train_array = input_array(train_data, var_leaf)
    test_array = input_array(test_data, var_leaf)
    input_shape = train_array.shape[1:]    
    model = cnn_function(input_shape, num_classes)
    model.summary()
    name = 'cnn'
elif var_vit:
    target_shape = (224,224,3)  #need to have square size
    train_array_1 = input_array(train_data, var_leaf)
    test_array_1 = input_array(test_data, var_leaf)
    #----------------------------------------------
    #train_array = np.repeat(train_array_1, 3, axis=-1)
    #test_array = np.repeat(test_array_1, 3, axis=-1) 
    #input_shape = train_array.shape[1:]
    #output_train = pad_crop_image_vit(train_array, target_shape, input_shape, sr)
    #output_test = pad_crop_image_vit(test_array, target_shape,input_shape, sr)
    #----------------------------------------------
    train_rgb = mono_to_color(train_array_1)
    train_array = ViT_resize_image(train_rgb, target_shape[0], target_shape[1])
    
    test_rgb = mono_to_color(test_array_1)
    test_array = ViT_resize_image(test_rgb, target_shape[0], target_shape[1])
  
    output_train_file = os.path.join(dataset_folder_helper, 'output_train.npy')
    np.save(output_train_file, train_array)
    
    output_test_file = os.path.join(dataset_folder_helper, 'output_test.npy')
    np.save(output_test_file, test_array)
    
    #------------------------------------------------
    input_shape = train_array.shape[1:]
    model = vision_transformer(input_shape[0], num_classes, patch_size = 32)
    model.summary()
    name = 'ViT'
else:
    train_array = input_array(train_data, var_leaf)
    test_array = input_array(test_data, var_leaf)
    train_array = np.repeat(train_array, 3, axis=-1)
    test_array = np.repeat(test_array, 3, axis=-1)
    input_shape = train_array.shape[1:]
    if var_resnet:
        model = resnet50(input_shape)          #input_shape = (128,431,1)
        model.summary() 
        name = 'resnet'                       #leaf_input_shape = (40,100,1)
    elif var_inception:                        #input_shape = (128,87,1) (chunk)
        model = inceptionv3(input_shape)
        model.summary()
        name = 'inception'
    elif var_xception:
        model = xception(input_shape)
        model.summary()
        name = 'xception'    
#-------------------------------------------------------------------------------
#earlystopping = callbacks.EarlyStopping(monitor = "val_loss", mode= "min",
                                        #patience= 10, restore_best_weights= True)

scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                        patience=5, min_lr=1e-8, verbose=1)

checkpointer = callbacks.ModelCheckpoint(filepath = dataset_folder_final + '/'+ dataset_name + '_model_' + name + '.hdf5',
                                         verbose=1, save_best_only=True)
#---------------------------------  Training -----------------------------------
print(f'================= Training will start with {KFOLD}-fold cross validation and {EPOCH} epochs ========================')
data_kfold, model_history = kfold_training(train_array, test_array,
                                           y_train_one_hot, y_test_one_hot,
                                           test_count, model, scheduler,
                                           checkpointer, k_fold = KFOLD,
                                           Batch_size= BATCH, num_epochs = EPOCH, var_leaf= var_leaf)
# -------------------------------- saving models -------------------------------
with open(dataset_folder_final+'/modelhistory_'+ dataset_name + name + '.pkl', 'wb') as f:            # Save model_history
    pickle.dump(model_history, f)
    
with open(dataset_folder_final+ '/data_kfold_'+ dataset_name + name + '.pkl', 'wb') as f:
    pickle.dump(data_kfold, f)                                                               # Save data_kfold for prediction

#------------------------------- Loading models --------------------------------
with open(dataset_folder_final + '/modelhistory_'+ dataset_name + name + '.pkl', 'rb') as f:                                      # Load model_history
    loaded_model_history = pickle.load(f)
    
with open(dataset_folder_final + '/data_kfold_'+ dataset_name + name + '.pkl', 'rb') as f:                                        # Load data_kfold for prediction
    data_kfold = pickle.load(f)
#---------------------------- Plotting learning curves -------------------------
plot_each_fold(model_history, dataset_folder_plots)
plot_avg_fold(model_history, dataset_folder_plots)

#-------------------------------------------------------------------------------
predicted_labels = label_prediction(data_kfold)
true_labels = y_test_encoded

np.save(dataset_folder_final+'/predicted_labels_'+ dataset_name + name + '.npy', np.array(predicted_labels))
np.save(dataset_folder_final+'/true_labels_bigmass_'+ dataset_name + name + '.npy', np.array(true_labels))

#-------------------------------------------------------------------------------
confusion_mat(true_labels, predicted_labels, dataset_folder_plots)
print("================ Confusion matrix has been plotted successfully ==================")
classification_reports(true_labels, predicted_labels, dataset_folder_plots)
print("================ Classification reports has been saved successfully ==============")
roc_curve_function(true_labels, predicted_labels, dataset_folder_plots)
print("================ ROC curve has been plotted successfully =========================")
print ("=============================== FINISH ===========================================")
#-------------------------------------------------------------------------------
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