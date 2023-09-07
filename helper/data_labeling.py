import pandas as pd
import shutil
import os

data = pd.read_csv('C:/Users/zahra/VoiceColab/dataset/metadata/data.csv',usecols=['MRN', 'Diagnosis'])
mrn_na= data['MRN'].isna().sum()
diagnosis_na= data['Diagnosis'].isna().sum()
data.dropna(axis=0,inplace=True)
data['label'] = ['Positive' if x=='Mass' else 'Negative' for x in data['Diagnosis']]
labels = 'Positive Negative'.split()



def datalabeling(FolderPath, VoicePath):
    
    if not os.path.exists(FolderPath):
        os.makedirs(FolderPath)
  
    a= 0    
    b= 0    
    c= 0    
    NotMatched=[]
    for filename in os.listdir(VoicePath):
        voicename = filename
        a+=1
        k=0
        for z in range(len(data['MRN'])):
            #if (voicename.split('-',1)[1].split('.')[0] == data['MRN'].iloc[z] or (voicename.split('-',1)[1].split('.')[0]).split('-')[0] == data['MRN'].iloc[z]) :    # for sustained_e
            if (voicename.split('.')[0] == data['MRN'].iloc[z] or voicename.split('-')[0] == data['MRN'].iloc[z] ):   # for other datasets
                g = data['label'].iloc[z]
                if not os.path.isdir(FolderPath +'/'+ g):
                    os.mkdir(FolderPath +'/'+ g)
                shutil.copy(VoicePath + '/'+ voicename, FolderPath +'/'+ g)  
                b+=1 
                k=1
                print('copied with z:', z)
        if k==0:
            var = voicename   
            NotMatched.append(var)
            c+=1    
            
if __name__ == '__main__':         
            
    VoicePath= 'C:/Users/zahra/VoiceColab/dataset/1_e/ConcatenatedAudio/concatenated_audio'
    FolderPath= 'C:/Users/zahra/VoiceColab/dataset/1_e/ConcatenatedAudio/labeled_audio_concat'
 
    datalabeling(FolderPath, VoicePath)            
            