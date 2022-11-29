import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import resample

def separate_sides(pi_data):
    left_df = pi_data[pi_data['foot'] == 'left']
    right_df = pi_data[pi_data['foot'] == 'right']

    left_data = pi_reshape(left_df)
    right_data = pi_reshape(right_df)
    return left_data, right_data

def pi_reshape(side_df):
    """
    reshapes continuous df to 3D array [31 x 11 x number of frames]
    :param side_df: continuous df
    :return: 3d array [31 x 11 x number of frames]
    """
    frames = side_df['frame'].unique()
    side_array = np.empty([31,11,frames[-1]])
    for i in frames:
        side_array[:, :, i - 1] = side_df.loc[side_df['frame'] == i, '0':'10'].values
    return side_array

def pi_force(array3):
    return np.sum(np.sum(array3, axis=0), axis=0)

def pi_step_segmentation(force_left, force_right, sampling_rate_pi):
    """
    :param force_left:
    :param force_right:
    :param sampling_rate_pi:
    :return:
    """
    IC_left = pi_ic(force_left, sampling_rate_pi)
    IC_right = pi_ic(force_right, sampling_rate_pi)
    TO_left = pi_to(force_left, IC_left)
    TO_right = pi_to(force_right, IC_right)
    events_dict = {'IC_left': np.array(IC_left[:-1]),
              'IC_right': np.array(IC_right[:-1]),
              'TO_left': np.array(TO_left),
              'TO_right': np.array(TO_right)
    }
    return events_dict

def pi_ic(force_side, sampling_rate):
    # calculate rate of force development (1st derivative of force)
    rfd = np.diff(force_side, n=1)*sampling_rate   # unit N/s

    # find where rfd > 1500 N/s based on Seiberl et al. (2018)
    rfd_1500 = np.where(rfd > 1500)[0]

    # first IC is when rfd first exceeds 1500 N/s
    IC_side = [rfd_1500[0]]

    # next IC of the same side has to be at least 250 ms away (fs/4)
    for i in range(1, len(rfd_1500)):
        if rfd_1500[i] - rfd_1500[i - 1] > sampling_rate / 4:
            IC_side.append(rfd_1500[i])
    IC_side=np.array(IC_side, dtype=int)
    return IC_side

def pi_to(force_side, IC_side):
    # filter force data (below 20 N threshold --> 0)
    force_side[force_side < 20] = 0
    TO_side=[]

    # loop to find Toe Off after respective IC
    # force = 0 for the first time after IC
    # clip before last IC (there might not be a toe off after)
    for IC in IC_side[:-1]:
        TO_side.extend(np.argwhere(force_side[IC+1:] == 0)[0] + IC+1)
    return TO_side

def pi_temporal_parameters(pi_events_dict, sampling_rate_pi):
    """
    calculates Ground Contact Time, Flight Time, Stride Time
    :param pi_events_dict: Dict containing all IC and TO events
    :param sampling_rate_pi: sampling rate fs
    :return temp_params_df: df with GCT, Flight T., Stride T., Side for all Steps
    """
    side = []
    gct = []
    st = []
    ft_left_off = []
    ft_right_off = []
    ft = []

    st_left = np.diff(pi_events_dict['IC_left']/sampling_rate_pi, n=1)
    st_right = np.diff(pi_events_dict['IC_right']/sampling_rate_pi, n=1)
    gct_left = (pi_events_dict['TO_left']-pi_events_dict['IC_left'])/sampling_rate_pi
    gct_right = (pi_events_dict['TO_right']-pi_events_dict['IC_right'])/sampling_rate_pi

    # 1st step left or right?
    if pi_events_dict['IC_left'][0] < pi_events_dict['IC_right'][0]:
        order = ['left', 'right']
        # TODO: somehow not working yet...
        for i in range(len(pi_events_dict['TO_left'])-1):
            ft_left_off.append((pi_events_dict['IC_right'][i]-pi_events_dict['TO_left'][i]) / sampling_rate_pi)
        for i in range(len(pi_events_dict['TO_right'])):
            ft_right_off.append((pi_events_dict['IC_left'][i+1] - pi_events_dict['TO_right'][i]) / sampling_rate_pi)


          #  ft_right_off.append((pi_events_dict['IC_left'][i+1]-pi_events_dict['TO_right'][i])/sampling_rate_pi)
          #  ft.append((pi_events_dict['IC_right'][i]-pi_events_dict['TO_left'][i])/sampling_rate_pi)
          #  ft.append((pi_events_dict['IC_right'][i] - pi_events_dict['TO_left'][i])/sampling_rate_pi)

  #  else:
     #   None

    IC_left = pi_events_dict['IC_left']
    temp_params_df = None
    return temp_params_df

def pi_separate_steps(pi_left, pi_right, pi_events):
    """
    separates the Stance phases for each step and save in dict
    :param pi_left: pressure data left foot 3d array shape [31 x 11 x frames]
    :param pi_right: pressure data right foot 3d array shape [31 x 11 x frames]
    :param pi_events: dict with IC and TO indices for each side
    :return pi_steps: dict with data for each step (stance phase) - values 3d array[31 x 11 x [IC:TO]]
    """
    # init dict
    pi_steps = {}

    # left side
    for i in range(len(pi_events['IC_left'])):
        pi_steps['left_'+str(i)]=pi_left[:,:,pi_events['IC_left'][i]:pi_events['TO_left'][i]]
    # right side
    for i in range(len(pi_events['IC_right'])):
        pi_steps['right_'+str(i)]=pi_right[:,:,pi_events['IC_right'][i]:pi_events['TO_right'][i]]

    return pi_steps

def pi_get_cop(pi_steps):
    """
    calculates COP coordinates for all steps
    :param pi_steps: dict with all steps (stance), 3d array with shape [31 x 11 x (IC:TO)]
    :return cop_dict: dict with COP coordinates as pd.DataFrame (columns = [x, y]) for each step
    """
    # init dict
    cop_dict = {}

    # loop over all steps
    for key in pi_steps:
        # init array for coordinates
        x = np.empty(pi_steps[key].shape[2])
        y = np.empty(pi_steps[key].shape[2])

        # calculate center of mass for each frame
        for frame in range(pi_steps[key].shape[2]):
            y[frame], x[frame]  = ndimage.measurements.center_of_mass(pi_steps[key][:,:,frame])

        # save COP coordinates (time series) in data frame) & append to dict
        cop_dict[key]=pd.DataFrame([x,y], index = ['x', 'y']).T

    return cop_dict

def pi_get_fpa(pi_steps, sensor_size):
    """
    calculates Force, Pressure & Contact Area Time Series for each step
    :param pi_steps: dict with all steps (stance), 3d array with shape [31 x 11 x (IC:TO)]
    :return fpa: dict with Force, Pressure, Area TS as pd.DataFrame (columns = [Force, Area, Pressure]) for each step
    """
    # init dict
    fpa_dict = {}

    # loop over all steps
    for key in pi_steps:
        # init arrays for F, A
        f = np.empty(pi_steps[key].shape[2])
        a = np.empty(pi_steps[key].shape[2])

        # calculate parameters for each frame
        for frame in range(pi_steps[key].shape[2]):
            # Area: count all non-zero sensors, multiply by sensor size
            a[frame] = np.count_nonzero(pi_steps[key][:,:,frame])*sensor_size
            # Force: sum up all pressure values
            f[frame] = np.sum(pi_steps[key][:,:,frame])
            # pressure = F/A when A=!0
            if a[frame] == 0:
                p[frame] = 0
            else:
                p[frame] = f[frame] / a[frame]


        # save Parameters  (time series) in data frame) & append to dict
        fpa_dict[key] = pd.DataFrame([f, p, a], index=['Force', 'Pressure', 'Area']).T

    return fpa_dict

def pi_dict_resample(input_dict, output_length):
    """
    Resamples Parameters for each Step at a given length into Multi-index DF
    :param input_dict: dict with Parameters as pd DataFrame for each step
    :param output_length: integer with desired output length
    :return df_resampled: Multi-Index DF, first level: parameters, second level: steps
    """
    # get list with steps
    steps = list(input_dict.keys())
    # get list with parameters
    parameters = list(list(input_dict.values())[0].columns)

    # resample dict via dict comprehension and re-create pd DataFrame as Dict Values
    dict_resampled = {key:pd.DataFrame(resample(value,output_length), columns= value.columns) for (key,value) in input_dict.items()}

    # initialize nested dict
    nested_dict = {}
    # extract each parameter in separate dict & nest
    for i in range(len(parameters)):
        nested_dict[parameters[i]] = {key: value.iloc[:,i] for (key,value) in dict_resampled.items()}

    # create multiindex df
    dict_of_df = {k: pd.DataFrame(v) for k, v in nested_dict.items()}
    df_resampled = pd.concat(dict_of_df, axis = 1)

    return df_resampled


#### Foot Segmentation
# Segmentation in 6 Segments (Medial-Lateral, Forefoot, Midfoot, Rearfoot)
def foot_segmentation(pressure_sp):
    forefoot = pressure_sp[0:11, :, :, :]
    midfoot = pressure_sp[11:21, :, :, :]
    rearfoot = pressure_sp[21:, :, :, :]

    if rearfoot[:, 0:6].mean() > rearfoot[:, 6:].mean():
        # if left pressure is bigger than right --> left foot --> needs to be flipped
        forefoot = np.fliplr(forefoot)
        midfoot = np.fliplr(midfoot)
        rearfoot = np.fliplr(rearfoot)

    def med_lat_slice(segment):
        med = segment[:, 0:6, :, :]
        lat = segment[:, 6:, :, :]
        return med, lat

    ffm, ffl = med_lat_slice(forefoot)
    mfm, mfl = med_lat_slice(midfoot)
    rfm, rfl = med_lat_slice(rearfoot)

    slices = {'FF_Med': ffm,
              'FF_Lat': ffl,
              'MF_Med': mfm,
              'MF_Lat': mfl,
              'RF_Med': rfm,
              'RF_Lat': rfl}

    return slices