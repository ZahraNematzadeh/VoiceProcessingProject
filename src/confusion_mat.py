import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, 
                             recall_score, f1_score,
                             roc_auc_score)

def confusion_mat(true_labels, predicted_labels, dataset_folder_plots):

    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    num_classes = len(confusion_mat)
    plt.figure(figsize=(8, 6))
    ax = plt.imshow(confusion_mat, interpolation='nearest', cmap='Blues')

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, confusion_mat[i, j], ha='center', va='center', color='black', fontsize=12)

    mappable = None
    for child in ax.get_children():
        if isinstance(child, plt.cm.ScalarMappable):
            mappable = child
            break

    if mappable is not None:
        #plt.colorbar(mappable)
        colorbar = plt.colorbar(mappable)
        colorbar.ax.tick_params(labelsize=10, pad=0.15)
        
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(len(confusion_mat)), np.arange(len(confusion_mat)))
    plt.yticks(np.arange(len(confusion_mat)), np.arange(len(confusion_mat)))
    plt.savefig(os.path.join(dataset_folder_plots, 'confusion_matrix_plot.png'))
    plt.close()
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    specificity = confusion_mat[0, 0]/ (confusion_mat[0, 0] + confusion_mat[0, 1])
    auc = roc_auc_score(true_labels, predicted_labels)
    fscore = f1_score(true_labels, predicted_labels)


    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC', 'F-score'],
        'Value': [accuracy, precision, recall, specificity, auc, fscore]
    })
    
    excel_file_path = os.path.join(dataset_folder_plots, 'evaluation_results.xlsx')
    results_df.to_excel(excel_file_path, index=False)
    
    print("Confusion Matrix:", confusion_mat)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("AUC:", auc)
    print('f-score : ', fscore)