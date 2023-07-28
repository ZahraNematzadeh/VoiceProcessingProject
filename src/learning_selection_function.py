from src.make_dataset_folder import make_dataset_folder
from src.get_dataset_name import get_dataset_name

def learning_selection_function(folder_path_train, folder_path_test, var_leaf):
    
    helper_output_path = '/content/drive/My Drive/VoiceProcessingProject_Outputs/HelpersOutputs'
    final_output_path = '/content/drive/My Drive/VoiceProcessingProject_Outputs/FinalOutputs'
    plots_output_path = '/content/drive/My Drive/VoiceProcessingProject_Outputs/Plots'
   
    if var_leaf == False:
        learning_selection = input("Enter 'c' for CNN or 't' for Transfer-Learning: ")
        if learning_selection.lower() == 'c':
            
            dataset_name = get_dataset_name(folder_path_train, -5)
            dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, 
                                                         visualizing='Melspectrogram', learning_name='CNN')
            dataset_folder_final = make_dataset_folder (final_output_path, dataset_name,
                                                        visualizing='Melspectrogram', learning_name='CNN')
            dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name,
                                                        visualizing='Melspectrogram', learning_name='CNN')
        else: 
            transfer_learning = input("Enter 'r' for Resnet50 or 'i' for InceptionV3 or 'x' for Xception: ")
            if transfer_learning.lower() == 'r':
                dataset_name = get_dataset_name(folder_path_train, -4)
                dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='Resnet50')
                dataset_folder_final = make_dataset_folder (final_output_path, dataset_name, visualizing='Melspectrogram', learning_name='Resnet50')
                dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Melspectrogram',learning_name='Resnet50')
               
            elif transfer_learning.lower() == 'i':
               
                dataset_name = get_dataset_name(folder_path_train, -4)
                dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='InceptionV3')
                dataset_folder_final = make_dataset_folder (final_output_path, dataset_name,visualizing='Melspectrogram', learning_name='InceptionV3')
                dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Melspectrogram',learning_name='InceptionV3')
                
            elif transfer_learning.lower() == 'x':
                 
                 dataset_name = get_dataset_name(folder_path_train, -4)
                 dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Melspectrogram', learning_name='Xception')
                 dataset_folder_final = make_dataset_folder (final_output_path, dataset_name, visualizing='Melspectrogram', learning_name='Xception')
                 dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Melspectrogram', learning_name='Xception')
                 
    elif var_leaf == True:
        learning_selection = input("Enter 'c' for CNN or 't' for Transfer-Learning: ")
        if learning_selection.lower() == 'c':
            
            dataset_name = get_dataset_name(folder_path_train,-5)
            dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Leaf', learning_name='CNN')
            dataset_folder_final = make_dataset_folder (final_output_path, dataset_name, visualizing='Leaf', learning_name='CNN')
            dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Leaf', learning_name='CNN')
            
        else:
            transfer_learning = input("Enter 'r' for Resnet50 or 'i' for InceptionV3 or 'x' for Xception: ")
            if transfer_learning.lower() == 'r':
               
                dataset_name = get_dataset_name(folder_path_train,-5)
                dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Leaf', learning_name='Resnet50')
                dataset_folder_final = make_dataset_folder (final_output_path, dataset_name, visualizing='Leaf', learning_name='Resnet50')
                dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Leaf', learning_name='Resnet50')
                
            elif transfer_learning.lower() == 'i':
                
                dataset_name = get_dataset_name(folder_path_train,-5)
                dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Leaf', learning_name='InceptionV3')
                dataset_folder_final = make_dataset_folder (final_output_path, dataset_name, visualizing='Leaf', learning_name='InceptionV3')
                dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Leaf', learning_name='InceptionV3')
                
            elif transfer_learning.lower() == 'x':
                
                dataset_name = get_dataset_name(folder_path_train,-5)
                dataset_folder_helper = make_dataset_folder (helper_output_path, dataset_name, visualizing='Leaf', learning_name='Xception')
                dataset_folder_final = make_dataset_folder (final_output_path, dataset_name, visualizing='Leaf', learning_name='Xception')
                dataset_folder_plots = make_dataset_folder (plots_output_path, dataset_name, visualizing='Leaf', learning_name='Xception')
                
        
        return dataset_folder_helper, dataset_folder_final, dataset_folder_plots, dataset_name