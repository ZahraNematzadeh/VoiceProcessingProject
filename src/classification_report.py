import os
import pandas as pd
from sklearn.metrics import classification_report

def classification_reports(true_labels, predicted_labels, dataset_folder_plots):

    classif_report = pd.DataFrame.from_dict(classification_report(true_labels, predicted_labels, output_dict=True))
    excel_file_path = os.path.join(dataset_folder_plots, 'evaluation_results.xlsx')
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
        classif_report.to_excel(writer, sheet_name='classification_report')

    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))



    
