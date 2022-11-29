# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:19:20 2022

@author: b1090197
"""

#import glob
import numpy as np
#import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import resample

#%% import
pi = pd.read_csv(r'PI1_03.csv')
pi.drop(columns='Unnamed: 0', inplace = True)
imu = pd.read_csv(r'IMU_03.txt', sep='\t', header=1)
fs_imu = 2000
fs_pi = 150

#%% PI preprocessing --> 3d array
pi['frame']=pi['frame']-2258
frames_pi=pi['frame'].unique()
left=pi[pi['foot']=='left']
right=pi[pi['foot']=='right']
pi_left=np.empty([31,11,frames_pi[-1]])
pi_right=np.empty([31,11,frames_pi[-1]])
for i in frames_pi:
    pi_left[:,:,i-1]=left.loc[left['frame']==i,'0':'10'].to_numpy()
    pi_right[:,:,i-1]=right.loc[right['frame']==i,'0':'10'].to_numpy()

#%% get IC - PI
rfd_left = np.diff(np.sum(np.sum(pi_left, axis=0), axis=0), n=1)*fs_pi   # unit N/s
rfd_right = np.diff(np.sum(np.sum(pi_right, axis=0), axis=0), n=1)*fs_pi
rfd_1500_r = np.where(rfd_right>1500)[0]
IC_right_pi = [rfd_1500_r[0]]
for i in range(1,len(rfd_1500_r)):
    if rfd_1500_r[i]-rfd_1500_r[i-1]>fs_pi/4:
        IC_right_pi.append(rfd_1500_r[i])
rfd_1500_l = np.where(rfd_left>1500)[0]
IC_left_pi= [rfd_1500_l[0]]
for i in range(1,len(rfd_1500_l)):
    if rfd_1500_l[i]-rfd_1500_l[i-1]>fs_pi/4:
        IC_left_pi.append(rfd_1500_l[i])
IC_left_pi = np.array(IC_left_pi)
IC_right_pi = np.array(IC_right_pi)
#%% get IC - IMU
r = np.linalg.norm(imu[imu.columns[imu.columns.str.contains(pat = 'R.Right.Foot_ImuA')]], axis = 1)
l = np.linalg.norm(imu[imu.columns[imu.columns.str.contains(pat = 'L.Left.Foot_ImuA')]], axis = 1)
IC_right_imu, _ = find_peaks(r, distance = 500, height = 5)
IC_left_imu, _ = find_peaks(l, distance = 500, height = 5)

#%% downsample IMU
imu_down = pd.DataFrame(resample(imu, frames_pi[-1]), columns = imu.columns)
res_left = np.linalg.norm(imu_down[imu_down.columns[imu_down.columns.str.contains(pat = 'R.Right.Foot_ImuA')]], axis = 1)
res_right = np.linalg.norm(imu_down[imu_down.columns[imu_down.columns.str.contains(pat = 'L.Left.Foot_ImuA')]], axis = 1)

#%%
force_left = np.sum(np.sum(pi_left, axis=0), axis=0)
force_right = np.sum(np.sum(pi_right, axis=0), axis=0)

#%% clip from first to last peak
force_left = force_left[IC_left_pi[0]:IC_left_pi[-1]+1]
imu_left = imu.iloc[IC_left_imu[0]:IC_left_imu[-1]]
imu_down_left = pd.DataFrame(resample(imu_left, len(force_left)), columns = imu.columns)
res_left = np.linalg.norm(imu_down_left[imu_down_left.columns[imu_down_left.columns.str.contains(pat = 'L.Left.Foot_ImuA')]], axis = 1)
#%%
#
t=IC_left_pi-IC_left_pi[0]
fig, ax1 = plt.subplots()
ax1.plot(force_left, color = 'r')
ax1.plot(t, force_left[t], 'x')
ax2 = ax1.twinx()
ax2.plot(res_left)
#ax2.plot(IC_left_imu-IC_left_imu[0], res_left[IC_left_imu-IC_left_imu[0]], 'x', color = 'blue')
#%%

#plt.plot(IC_left_imu, force_left[IC_left_imu], 'x', color='blue')
#plt.plot(IC_left_pi, force_left[IC_left_pi], 'x', color='g')
