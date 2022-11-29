# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:00:02 2022

@author: b1090197
"""

import glob
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import resample

#%%
#path = r'D://Salzburg//TexSense//Messungen//IMU//treadmill_finki'
files = glob.glob('*.txt')
files.sort()
#%% import function
def import_file(path):
    return pd.read_csv(path, sep='\t', header=1)

#%% IC finder
def IC_imu(df):
    r = np.linalg.norm(df[df.columns[df.columns.str.contains(pat = 'R.Right.Foot_ImuA')]], axis = 1)
    l = np.linalg.norm(df[df.columns[df.columns.str.contains(pat = 'L.Left.Foot_ImuA')]], axis = 1)
    
    #b, a = butter(4, 100/2000, btype='low')
    #r = filtfilt(b, a, right_foot_racc)
    #l = filtfilt(b, a, left_foot_racc)
    
    IC_right, _ = find_peaks(r, distance = 500, height = 5)
    IC_left, _ = find_peaks(l, distance = 500, height = 5)
    
    if r.shape[0]-IC_right[-1]<800:
        IC_right = IC_right[:-1]
    if l.shape[0]-IC_left[-1]<800:
        IC_left = IC_left[:-1]
       
    print('number of steps L:', len(IC_left)-1)
    print('number of steps R:', len(IC_right)-1)
    
    #if len(IC_left) != len(IC_right):        
     #   if len(IC_right)<len(IC_left):
      #      IC_left=IC_left[:len(IC_right)]
       # else:
        #    IC_right = IC_right[:len(IC_left)]
            
    return IC_right, IC_left

#%% print GCT
def print_gct(IC_right, IC_left, TO_right, TO_left):
    print('Stance Times [ms]')
    stance_l = []
    stance_r = []
    for i in range(len(IC_left)):
        stance_l.append(TO_left[i]-IC_left[i])
        stance_r.append(TO_right[i]-IC_right[i])     
    
    print('Min Left:', np.min(stance_l)/2)
    print('Min Right:', np.min(stance_r)/2)
    print('Max Left:', np.max(stance_l)/2)
    print('Max Right:', np.max(stance_r)/2)
    return

#%% TO finder
def TO_imu(GyroY, IC):
    ### try find IC + 400ms
    TO=[]
    sig = GyroY.values
    for i in range(len(IC)):
        snip = sig[IC[i]+100:IC[i]+800] # search window + 400 ms
        peak, _ = find_peaks(snip, distance = 600, height = 200)
        TO.append(list(peak+IC[i]+100+48)) # add bias + 0.024 s
    
    return sum(TO, [])

#%% events
def events(df):
    IC_right, IC_left = IC_imu(df)
    TO_right = TO_imu(df['R.Right.Foot_Gyro :Y(D/s):'], IC_right)
    TO_left = TO_imu(df['L.Left.Foot_Gyro :Y(D/s):'], IC_left)
    
    print_gct(IC_right, IC_left, TO_right, TO_left)
    
    return IC_right, IC_left, TO_right, TO_left

#%% Skript starts
############################################
############################################
df = import_file(files[0])
IC_right, IC_left = IC_imu(df)

r = np.linalg.norm(df[df.columns[df.columns.str.contains(pat = 'R.Right.Foot_ImuA')]], axis = 1)
l = np.linalg.norm(df[df.columns[df.columns.str.contains(pat = 'L.Left.Foot_ImuA')]], axis = 1)

#%%
plt.plot(r)
plt.plot(IC_right, r[IC_right], 'x', color = 'green')
#%%
plt.plot(l, color = 'orange')
plt.plot(IC_left, l[IC_left], 'x', color = 'red')
#%%
print((IC_left[0]-IC_right[0])/2000)
print((IC_left[-1]-IC_right[-1])/2000)