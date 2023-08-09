import random
import librosa
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.signal import savgol_filter

def get_informative_chunk(wave_data, sr, var_leaf):
    
    final_chunk = []

    for data in wave_data:
          signal = data[0]
          filename = data[1]
          label = data[2]
          
          if var_leaf==True:
              signal = signal[0]
          
          total_time= librosa.get_duration(y=signal, sr=sr)  
          time_stamp = np.linspace(start= 0, stop=total_time , num= len(signal))
          t = time_stamp 
      #------------- adding noise to the 500 points of signal
          sig = signal
          noisy_sig=[]
          noise_num= 500
          rand_range = 0.1*(max(signal))
          for i in range(noise_num):
              noise =random.uniform(-rand_range, rand_range)
              si = sig[i]+noise
              noisy_sig.append(si)
          for i in range (noise_num, len(signal)):    
              noisy_sig.append(signal[i])
      #-------------- getting avg of signal ---------------------------
          signalvoice2 = noisy_sig  
          avg_signal = (integrate.simps(signalvoice2,t))/ (t[len(t)-1]-t[0])    # getting avg signal
          zero_mean_signal = [] 
          for i in range(len(signalvoice2)):     #getting zero mean signal g(t)
              zero_mean = np.abs(signalvoice2[i] - avg_signal)
              zero_mean_signal.append(zero_mean)
      #--------------- orange line -------------------------------------
          intg_time_signal = []
          for i in range(len(t)):
              if i==0:
                  I = 0
                  intg_time_signal.append(I)
              else:    
                  I=((t[i-1]-t[0])*intg_time_signal[i-1]) + (((zero_mean_signal[i]+zero_mean_signal[i-1])/2)*(t[i]-t[i-1]))
                  I= I/(t[i]-t[0])
                  intg_time_signal.append(I)
      #--------------- Smoothing orange line ----------------------------
          smoothed_curve = savgol_filter(intg_time_signal, 2000, 3)
      #--------------- Min and Max --------------------------------------
          smth_val = smoothed_curve
          imax=[]
          imin=[]
          for i in range(len(smoothed_curve)):
              if i==0 and smth_val[i]>smth_val[i+1]:
                  max_ind=i
                  imax.append(max_ind)
              elif i==0 and smth_val[i]<smth_val[i+1]:
                  min_ind=i
                  imin.append(min_ind)
              elif i!=len(smoothed_curve)-1 and smth_val[i]> smth_val[i-1] and smth_val[i]>smth_val[i+1]:
                  max_ind=i
                  imax.append(max_ind)
              elif i!=len(smoothed_curve)-1 and smth_val[i]< smth_val[i-1] and smth_val[i]<smth_val[i+1]:
                  min_ind=i
                  imin.append(min_ind)
              elif i==len(smoothed_curve)-1 and smth_val[i]< smth_val[i-1]:
                  min_ind=i
                  imin.append(min_ind) 
      #-----------------------------  min and max ---------------------------
          diff_df = pd.DataFrame(columns=['diff', 'imin', 'imax'])
          diff_data = []
          for i in range(0, len(imax)):
              if imax[0]==0 and i!= len(imax)-1:
                  diff_row = {'diff': imax[i+1] - imin[i], 'imin': imin[i], 'imax': imax[i+1]}
              elif imin[0]==0:
                  diff_row = {'diff': imax[i] - imin[i], 'imin': imin[i], 'imax': imax[i]}
              diff_data.append(diff_row)
            
          diff_df = pd.DataFrame(diff_data)
          diff_df['diff'] = diff_df['diff'].astype('int')

          idx2 = diff_df['diff'].idxmax()
          tmin = diff_df['imin'].iloc[idx2]
      #----------------------------- selecting chunks -----------------------
          if tmin + sr > len(signal):
              t_add_right = np.abs(len(signal)- tmin)
              t_add_left = np.abs(sr - t_add_right)
              xx = tmin - t_add_left
              one_sec_chunk = signal[xx:]
              final_chunk.append((one_sec_chunk, filename, label))
          else:
              one_sec_chunk = signal[tmin: tmin + sr]
              final_chunk.append((one_sec_chunk, filename, label))
    return final_chunk 