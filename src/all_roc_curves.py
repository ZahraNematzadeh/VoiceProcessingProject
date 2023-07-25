from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def all_roc_curves(true_labels_original,predicted_labels_original,
               true_labels_bigmass,predicted_labels_bigmass,
               true_labels_smallmass,predicted_labels_smallmass,
               true_labels_mediummass,predicted_labels_mediummass):
    
    fpr_original, tpr_original, _ = roc_curve(true_labels_original, predicted_labels_original)
    fpr_bigmass, tpr_bigmass, _ = roc_curve(true_labels_bigmass, predicted_labels_bigmass)
    fpr_smallmass, tpr_smallmass, _ = roc_curve(true_labels_smallmass, predicted_labels_smallmass)
    fpr_mediummass, tpr_mediummass, _ = roc_curve(true_labels_mediummass, predicted_labels_mediummass)
    
    auc_original = roc_auc_score(true_labels_original, predicted_labels_original)
    auc_bigmass = roc_auc_score(true_labels_bigmass, predicted_labels_bigmass)
    auc_smallmass = roc_auc_score(true_labels_smallmass, predicted_labels_smallmass)
    auc_mediummass = roc_auc_score(true_labels_mediummass, predicted_labels_mediummass)
    
    plt.plot(fpr_original, tpr_original, label='Original Data  (AUC = {:.2f})'.format(auc_original))
    plt.plot(fpr_bigmass, tpr_bigmass, label='Big Mass Data (AUC = {:.2f})'.format(auc_bigmass))
    plt.plot(fpr_smallmass, tpr_smallmass, label='Small Mass Data (AUC = {:.2f})'.format(auc_smallmass))
    plt.plot(fpr_mediummass, tpr_mediummass, label='Medium Mass Data (AUC = {:.2f})'.format(auc_mediummass))
    
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()