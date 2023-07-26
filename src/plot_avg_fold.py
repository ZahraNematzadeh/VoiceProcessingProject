import os
import numpy as np
import matplotlib.pyplot as plt

def plot_avg_fold(model_history, dataset_name):
    
    if not os.path.exists('/content/drive/My Drive/VoiceProcessingProject_Outputs/Plots'):
        os.makedirs('/content/drive/My Drive/VoiceProcessingProject_Outputs/Plots')
    dataset_folder = os.path.join('/content/drive/My Drive/VoiceProcessingProject_Outputs/Plots', dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    avg_accuracy = np.mean([history.history['categorical_accuracy'] for history in model_history], axis=0)
    avg_accuracy_val = np.mean([history.history['val_categorical_accuracy'] for history in model_history], axis=0)
    avg_loss = np.mean([history.history['loss'] for history in model_history], axis=0)
    avg_loss_val = np.mean([history.history['val_loss'] for history in model_history], axis=0)
    
    plt.figure(figsize=(4.4, 2.8))
    plt.plot(avg_accuracy, 'g', label='Training Accuracy')
    plt.plot(avg_accuracy_val, 'b', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy across Folds')
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(dataset_folder, 'avg_accuracy_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()
    
    plt.figure(figsize=(4.4, 2.8))
    plt.plot(avg_loss, label='Training Loss')
    plt.plot(avg_loss_val, label='Validation Loss')
    plt.plot(avg_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss across Folds')
    plt.legend(fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.savefig(os.path.join(dataset_folder, 'avg_loss_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()
    print('=========================================================')
    print('=========================================================')
