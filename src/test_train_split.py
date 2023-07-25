import splitfolders

input_folder= 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/2_DataLabeling/CleanedData/FourLabels'
output= 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/2_DataLabeling/CleanedData/Test_Train/FourLabels'
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .2))