import os
import matplotlib.pyplot as plt

def plot_each_fold(model_history, dataset_name, dataset_folder_plots):
    
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
        plt.savefig(os.path.join(dataset_folder_plots, f'fold_{fold+1}_accuracy.png'))
        plt.close()
    
        plt.figure(figsize=(4.4, 2.8))
        plt.plot(loss, label='Training Loss - Fold {}'.format(fold+1))
        plt.plot(val_loss, label='Validation Loss - Fold {}'.format(fold+1))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss - Fold-wise')
        plt.legend()
        plt.savefig(os.path.join(dataset_folder_plots, f'fold_{fold+1}_loss.png'))
        plt.close()
    print('=========================================================')
    print('Loss and Accuracy for each fold have been plotted successfully. Please check the directory.')
    print('=========================================================')