from sklearn.metrics import classification_report

def classification_reports(true_labels, predicted_labels):
    print(classification_report(true_labels, predicted_labels))
