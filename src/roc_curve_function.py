from sklearn.metrics import (roc_curve, roc_auc_score)
import matplotlib.pyplot as plt


def roc_curve_function(true_labels, predicted_labels):
    
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)                 # Calculate the false positive rate, true positive rate, and thresholds
    auc = roc_auc_score(true_labels, predicted_labels)                              # Calculate the AUC (Area Under the Curve)
    
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')                                        # Plot the diagonal line for reference
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (AUC = {:.2f})'.format(auc))
    plt.legend(loc='lower right')
    plt.show()