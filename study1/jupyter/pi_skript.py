# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:00:48 2022

@author: lucas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
conditions = pd.read_csv('conditions.csv', sep = ';', index_col=0)

#%%
trial = '_03'
fs = conditions.loc[trial]['fs']
df = pd.read_csv(r'PI1'+trial+'.csv', index_col=0)
frames=df['frame'].unique()
#%%
d3 = df[(df['frame']>2258)]
f3 = d3['frame'].unique()
d3.to_csv('pi1_03.csv')

#%% seperate left-right
left=df[df['foot']=='left']
right=df[df['foot']=='right']

#%% seperate by frames (alternative: reshape? 31,11,frames[-1])

left_data=np.empty([31,11,frames[-1]])
right_data=np.empty([31,11,frames[-1]])

# test if format is complete
if np.shape(left.loc[:,'0':'10'].to_numpy())[0]%31 != 0 or np.shape(right.loc[:,'0':'10'].to_numpy())[0]%31 != 0:
    print('Dimensions dont fit, missing data?!')


for i in frames:
    left_data[:,:,i-1]=left.loc[left['frame']==i,'0':'10'].to_numpy()
    right_data[:,:,i-1]=right.loc[right['frame']==i,'0':'10'].to_numpy()
    
#%% force threshold 
force_left=np.sum(np.sum(left_data, axis=0), axis=0)
force_right=np.sum(np.sum(right_data, axis=0), axis=0)
#force_left[force_left < 200]=0
#force_right[force_right < 200]=0

#%% plot force data
plt.plot(force_left)
plt.plot(force_right)

#%% Rate of Force Development (detection)
rfd_left = np.diff(force_left, n=1)*fs   # unit N/s
rfd_right = np.diff(force_right, n=1)*fs # unit N/s

plt.plot(force_left)
plt.plot(rfd_right)
plt.hlines(1500,0,len(rfd_right), color='r')
plt.ylim([-5,1700])

#%% IC detection

# find points in time where rate of force development 1st exceeds 1500 N --> IC
# next IC of the same side has to be at least 250 ms away (fs/4)

rfd_1500_r = np.where(rfd_right>1500)[0]
IC_right = [rfd_1500_r[0]]
for i in range(1,len(rfd_1500_r)):
    if rfd_1500_r[i]-rfd_1500_r[i-1]>fs/4:
        IC_right.append(rfd_1500_r[i])
    
rfd_1500_l = np.where(rfd_left>1500)[0]
IC_left= [rfd_1500_l[0]]
for i in range(1,len(rfd_1500_l)):
    if rfd_1500_l[i]-rfd_1500_l[i-1]>fs/4:
        IC_left.append(rfd_1500_l[i])

#%%      
plt.plot(force_right)
plt.plot(IC_right, force_right[IC_right], 'x', color = 'green')
#%%
plt.plot(force_left, color = 'orange')
plt.plot(IC_left, force_left[IC_left], 'x', color = 'r')
#%%
print((IC_left[0]-IC_right[0])/fs)
print((IC_left[-1]-IC_right[-1])/fs)
#%% Separate Steps 
#retruns: 
    # Lists with start & stop frames
    # DF with Force for all steps (resampled)
    #   --> Force TS for all steps
    
def step_separator(force):
    f0=np.argwhere(force == 0)
    
    step_start=[]
    # step_start.extend(f0[0])
    for i in range(len(f0)-1):
        if (f0[i+1]-f0[i])>1:
            step_start.extend(f0[i])
    step_stop=[]
    # step_stop.extend(f0[0])
    for i in range(1, len(f0)):
         if abs(f0[i]-f0[i-1]) > 1:
                step_stop.extend(f0[i]+1)
                
    from scipy.signal import resample
    steps=np.empty([101,len(step_start)])
    
    for i in range(len(step_start)):
        x=force[step_start[i]:step_stop[i]]   # extracts steps
        steps[:,i]=(resample(x, 101))   # resamples steps to 101 frames
    steps[steps<0]=0
    # output steps as nice Dataframe
    steps = pd.DataFrame(data=steps)
    steps.columns=["step_"+str(i) for i in range(1, steps.shape[1] + 1)]

    return steps, step_start, step_stop

#%%
steps_right, start, stop = step_separator(force_right)

#%% Pressure Extracor

# requires:
    # start & stop lists (from step separator)
    # pressure data as 3d np array (31, 11, frames)

# returns:
    # 3d np array with mean pressure image for each step
    # 4d np array with pressure image for each step (resampled)
    #   shape: (31,11,101, #steps)
    #   --> pressure map TS for all steps

def steps_pressure(start, stop, data):
    
    from scipy.signal import resample
    steps_mean_pressure=np.zeros([31,11,len(start)])      # initialize Mean pressure array (Shape 31, 11, #steps)
    steps_standarized_pressure=np.zeros([31,11,101,len(start)])   # initialize standarized pressure array (Shape 31, 11, 101, #steps)
    
    for i in range(len(start)):      # for number of steps
        steps_mean_pressure[:,:,i]=(data[:,:,start[i]:stop[i]]).mean(axis=2)   #calculate mean pressure picture
        
        step_new=np.zeros([31,11,101])
        step_old=data[:,:,start[i]:stop[i]]
        step_new=resample(step_old, 101, axis=2)           # resample all pressure pictures of current step to 101
    
        steps_standarized_pressure[:,:,:,i]=step_new       # append to 4D array
    
    steps_mean_pressure[steps_mean_pressure<0]=0
    steps_standarized_pressure[steps_standarized_pressure<0]=0
    
    return steps_mean_pressure, steps_standarized_pressure

#%% pressure matrix

right_mp, right_sp = steps_pressure(start, stop, right_data)

f= right_sp[:,:,50,50]

#%% 
# Af = np.count_nonzero(f)*(.82**2)
# Pf = np.sum(f)/np.count_nonzero(f)
# Ff = Af*Pf
# Ff2= np.sum(f)*(.82**2)

#%% calculate Contact Area, Force, Pressure
# requries:
    # 4D array with standarized, step-separated pressure
    
# returns:
    # Contact Area DF
    # Force DF
    # Pressure DF for all steps


def A_F_P (steps_standarized_pressure):
    sensel=.82**2   # sensel size
    
    steps_standarized_pressure[steps_standarized_pressure < 1]=0  # filter all pressure <1 
    
    A=np.zeros([101, steps_standarized_pressure.shape[3]])
    F=np.zeros([101, steps_standarized_pressure.shape[3]])
    
    for i in range(steps_standarized_pressure.shape[3]):
        for j in range(101):
            temp=steps_standarized_pressure[:,:,j,i]    # current frame
            A[j,i]=np.count_nonzero(temp)*sensel        # count active sensels * sensel Area
            F[j,i]=np.sum(temp)*sensel                  # sum all sensels * sensel Area
    # output steps as nice Dataframe
    A = pd.DataFrame(data=A)
    A.columns=["step_"+str(i) for i in range(1, A.shape[1] + 1)]
    
    F = pd.DataFrame(data=F)
    F.columns=["step_"+str(i) for i in range(1, F.shape[1] + 1)]
    
    P=F/A      # broadcasting
            
    return A, F, P

#%% 
A, F, P = A_F_P(right_sp)


#%% calculate COP coordinates (TS) for each step

# Requires:
    # 4D Array with Pressures for each step
# Returns:
    # COP Coordinates TS for each step
def cop (steps_standarized_pressure):
    from scipy import ndimage
    x=np.empty((101,steps_standarized_pressure.shape[3]))    # shape [101, number of steps]
    y=np.empty((101,steps_standarized_pressure.shape[3]))
    for j in range(steps_standarized_pressure.shape[3]):
        for i in range(101):
            current_frame=steps_standarized_pressure[:,:,i,j]
            x[i,j]=(ndimage.measurements.center_of_mass(current_frame))[1]
            y[i,j]=(ndimage.measurements.center_of_mass(current_frame))[0]
            #print('i ', i, '   j ', j)
    return x,y

#%%
x,y = cop(right_sp)

# plot example COP Path Step 3
plt.imshow(np.mean(right_sp[:,:,:,2], axis = 2), cmap='jet', interpolation='nearest')
plt.scatter(x[:,2], y[:,2], marker='.', c=np.linspace(1,np.tan(len(y[:,2])),101))
