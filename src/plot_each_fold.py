import os
import matplotlib.pyplot as plt

def plot_each_fold(model_history, dataset_name):
    
    if not os.path.exists('/content/drive/My Drive/VoiceProcessingProject_Outputs/Plots'):
        os.makedirs('/content/drive/My Drive/VoiceProcessingProject_Outputs/Plots')
    dataset_folder = os.path.join('/content/drive/My Drive/VoiceProcessingProject_Outputs/Plots', dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    for fold, history in enumerate(model_history):
    
        accuracy = history.history['categorical_accuracy']
        val_accuracy = history.history['val_categorical_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
    
        plt.figure(figsize=(4.4, 2.8))
        plt.plot(accuracy, 'g', label='Training Accuracy - Fold {}'.format(fold+1))
        plt.plot(val_accuracy, 'b', label='Validation Accuracy - Fold {}'.format(fold+1))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy - Fold-wise')
        plt.legend()
        plt.savefig(os.path.join(dataset_folder, f'fold_{fold+1}_accuracy.png'))
        plt.close()
        #plt.show()
    
        plt.figure(figsize=(4.4, 2.8))
        plt.plot(loss, label='Training Loss - Fold {}'.format(fold+1))
        plt.plot(val_loss, label='Validation Loss - Fold {}'.format(fold+1))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss - Fold-wise')
        plt.legend()
        plt.savefig(os.path.join(dataset_folder, f'fold_{fold+1}_loss.png'))
        plt.close()
        #plt.show()
        print('=========================================================')
        print('=========================================================')