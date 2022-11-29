import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from study1.io.conditions import load_imu
from study1.utilities.imu import get_detection_windows
from study1.utilities.imu import k1, k3, k7, k8, k9, t1, t4, t8


# load example file
trial = 'D:\\Salzburg\\TexSense\\Messungen\\IMU\\treadmill_finki\\finki__17.txt'
imu_data = load_imu(trial)

# extract foot IMU data
left_foot = imu_data[imu_data.columns[imu_data.columns.str.contains(pat = 'L_Foot')]]
right_foot = imu_data[imu_data.columns[imu_data.columns.str.contains(pat = 'R_Foot')]]

# segment gait cycles
# 1. get dectection windows
windows_right = get_detection_windows(right_foot, 2000)
windows_left = get_detection_windows(left_foot, 2000)

# 2. loop over detection windows & caculate IC and TO after the proposed methods
# left side:
events_left = pd.DataFrame(data=None)
for i in range(len(windows_left['lower_IC'])):

    # IC Detection: slice current detection window
    ic_window = left_foot.loc[windows_left['lower_IC'][i]:windows_left['upper_IC'][i]]

    # k1 global min of angular velocity around Y
    events_left.loc[i, 'IC_1']=k1(ic_window)
    # k3 first min of angular velocity around Y < 100°/s
    events_left.loc[i, 'IC_3'] = k3(ic_window)
    # k7 global max of angular velocity norm
    events_left.loc[i, 'IC_7'] = k7(ic_window)
    # k8 global max of vertical acceleration (Z)
    events_left.loc[i, 'IC_8'] = k8(ic_window)
    # k9 global max of acceleration norm
    events_left.loc[i, 'IC_9'] = k9(ic_window)

    # TO Detection8 window
    to_window = left_foot.loc[windows_left['lower_TO'][i]:windows_left['upper_TO'][i]]

    # t1 global min of angular velocity around Y
    events_left.loc[i, 'TO_1'] = t1(to_window)
    # t4 global max of angular velocity norm
    events_left.loc[i, 'TO_4'] = t4(to_window)
    # t8 global max of acceleration norm
    events_left.loc[i, 'TO_8'] = t8(to_window)

# right side:
events_right = pd.DataFrame(data=None)
for i in range(len(windows_right['lower_IC'])):

    # IC Detection: slice current detection window
    ic_window = right_foot.loc[windows_right['lower_IC'][i]:windows_right['upper_IC'][i]]

    # k1 global min of angular velocity around Y
    events_right.loc[i, 'IC_1']=k1(ic_window)
    # k3 first min of angular velocity around Y < 100°/s
    events_right.loc[i, 'IC_3'] = k3(ic_window)
    # k7 global max of angular velocity norm
    events_right.loc[i, 'IC_7'] = k7(ic_window)
    # k8 global max of vertical acceleration (Z)
    events_right.loc[i, 'IC_8'] = k8(ic_window)
    # k9 global max of acceleration norm
    events_right.loc[i, 'IC_9'] = k9(ic_window)

    # TO Detection8 window
    to_window = right_foot.loc[windows_right['lower_TO'][i]:windows_right['upper_TO'][i]]

    # t1 global min of angular velocity around Y
    events_right.loc[i, 'TO_1'] = t1(to_window)
    # t4 global max of angular velocity norm
    events_right.loc[i, 'TO_4'] = t4(to_window)
    # t8 global max of acceleration norm
    events_right.loc[i, 'TO_8'] = t8(to_window)

