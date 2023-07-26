import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, 
                             recall_score, f1_score,
                             roc_auc_score)

def confusion_mat(true_labels, predicted_labels, dataset_name, dataset_folder_plots):

    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(4.4, 3.2))
    plt.imshow(confusion_mat, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(len(confusion_mat)), np.arange(len(confusion_mat)))
    plt.yticks(np.arange(len(confusion_mat)), np.arange(len(confusion_mat)))
    plt.savefig(os.path.join(dataset_folder, 'confusion_matrix_plot.png'))
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