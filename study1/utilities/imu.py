import numpy as np
import pandas as pd
from study1.utilities.signal import butter_2nd
from scipy.signal import find_peaks

def get_midswing(gyr_y_series, fs):
    """
    finds peaks in Angular Velocity around Y Axis of Foot IMU with distance of at least 0.5 s for the same side
    :param gyr_y_series: array or pd.Series with gyr_y signal
    :param fs: integer with sampling rate
    :return int array with mid_swing indices:
    """
    gyr_y_filt = butter_2nd(gyr_y_series, 2000, 90)
    mid_swing, _ = find_peaks(gyr_y_filt, distance = fs * 60 / 100)
    return mid_swing

def get_midstance(foot_imu_side, mid_swing_side):
    """
    finds midstance (global minimum in the Angular Velocity Norm of Foot IMU in the interval between Midswing and 40 % of the following Midswing
    :param foot_imu_side: pd.DataFrame with IMU Data of one Side
    :param mid_swing_side: int array with mid_swing indices
    :return mid_stance_side: int array with mid_stance indices
    """
    # foot gyro signals are stored at index [3:6] in the IMU df
    ang_norm = np.linalg.norm(foot_imu_side.iloc[:,3:6], axis = 1)

    # initialize empty list
    mid_stance_side = []

    # loop over all mid_swing events - 1
    # get minimum in ang_norm in the interval of:
    # mid_swing[i] to mid_swing[i] + 40 % of the distance to mid_swing[i+1]
    # add index of current mid_swing to the found mid_stance index and append to list
    for i in range(len(mid_swing_side) - 1):
        mid_stance_side.append(np.argmin(
            ang_norm[mid_swing_side[i]:mid_swing_side[i] + (2 * (mid_swing_side[i + 1] - mid_swing_side[i])) // 5]) +
                               mid_swing_side[i])

    return np.array(mid_stance_side)

def get_detection_windows(foot_imu_df, sampling_rate):
    """
    calculates the search windows for  IC and TO search
    :param foot_imu_df: DataFrame with Foot IMU for one Side
    :param sampling_rate: Sampling Rate - integer
    :return window_dict: dict with Lists of interval boundries for IC and TO search
    """

    # get midswing indices
    midswing = get_midswing(foot_imu_df.iloc[:,4], sampling_rate)

    # filter all Data at 30 Hz
    foot_df_filt = pd.DataFrame(butter_2nd(foot_imu_df, sampling_rate, 30), columns=foot_imu_df.columns)

    # get interval boundries based on Angular Velocity Signal
    ang_norm = np.linalg.norm(foot_df_filt.iloc[:,3:6], axis = 1)

    # init empty lists
    TO_end = []
    IC_start = []
    mid = []

    # for each step (between 2 midswing events):
    # find 3 peaks (1st = midstance, 2nd = TO upper boundry, 3rd = IC lower boundry (of the next step))
    for i in range(len(midswing) - 1):
        peaks, _ = find_peaks(-ang_norm[midswing[i]:midswing[i + 1]], prominence=150) # <-- 150 just a guess
        try:
            len(peaks)==3
        except:
            print('Error in window detection:'
                  'did not find 3 peaks in ang_norm signal between'
                  'midswing['+str(i)+'] and midswing['+str(i+1)+']')
        mid.append(peaks[0] + midswing[i])
        TO_end.append(peaks[1] + midswing[i])
        IC_start.append(peaks[2] + midswing[i])

    # clip first element of mid and TO_end & last element of IC_start to allign order
    mid = mid[1:]
    TO_end = TO_end[1:]
    IC_start = IC_start[:-1]

    window_dict = {'lower_IC': IC_start,
                   'upper_IC': mid,
                   'lower_TO': mid,
                   'upper_TO': TO_end}
    return window_dict

# IC_detection methods
def k1(slice_df):
    """
    global minimum of angular velocity around y
    :param slice_df: DataFrame slice of current interval for IC detection
    :return IC as int
    """
    return int(slice_df.iloc[:,4].idxmin())

def k3(slice_df):
    """
    first minimum of angular velocity around y < 100°/s
    :param slice_df: DataFrame slice of current interval for IC detection
    :return IC index as int
    """
    # find first peak lower than 100
    peaks, _ = find_peaks(-slice_df.iloc[:,4], height = -100)
    try:
        IC = int(peaks[0] + slice_df.index[0])
    except:
        print('could not find IC in k3')
        IC = None
    return IC

def k7(slice_df):
    """
    global maximum of angular velocity norm
    :param slice_df: DataFrame slice of current interval for IC detection
    :return IC index as int
    """
    return int(np.argmax(np.linalg.norm(slice_df.iloc[:,3:6], axis = 1)) + slice_df.index[0])

def k8(slice_df):
    """
    global maximum of vertical acceleration
    :param slice_df: DataFrame slice of current interval for IC detection
    :return IC index as int
    """
    return int(slice_df.iloc[:,2].idxmax())

def k9(slice_df):
    """
    global maximum of acceleration norm
    :param slice_df: DataFrame slice of current interval for IC detection
    :return IC index as int
    """
    return int(np.argmax(np.linalg.norm(slice_df.iloc[:,:3], axis = 1)) + slice_df.index[0])

def t1(slice_df):
    """
    global minimum of angular velocity around y
    :param slice_df: DataFrame slice of current interval for IC detection
    :return IC index as int
    """
    return k1(slice_df)

def t4(slice_df):
    """
    global maximum of angular velocity norm
    :param slice_df: DataFrame slice of current interval for IC detection
    :return IC index as int
    """
    return k7(slice_df)

def t8(slice_df):
    """
        global maximum of angular velocity norm
        :param slice_df: DataFrame slice of current interval for IC detection
        :return IC index as int
        """
    return k9(slice_df)










