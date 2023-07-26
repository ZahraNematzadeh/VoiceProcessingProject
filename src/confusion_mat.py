import os
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, 
                             recall_score, f1_score,
                             roc_auc_score)

def confusion_mat(true_labels, predicted_labels, dataset_name):

    if not os.path.exists('/content/drive/My Drive/VoiceProcessingProject_Outputs/Plots'):
        os.makedirs('/content/drive/My Drive/VoiceProcessingProject_Outputs/Plots')
    dataset_folder = os.path.join('/content/drive/My Drive/VoiceProcessingProject_Outputs/Plots', dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_mat)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(f'/content/drive/My Drive/VoiceProcessingProject_Outputs/Plots/{dataset_name}/confusion_matrix_plot.png')
    plt.close()
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    specificity = confusion_mat[0, 0]/ (confusion_mat[0, 0] + confusion_mat[0, 1])
    auc = roc_auc_score(true_labels, predicted_labels)
    fscore = f1_score(true_labels, predicted_labels)
    
    print("Confusion Matrix:", confusion_mat)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("AUC:", auc)
    print('f-score : ', fscore)