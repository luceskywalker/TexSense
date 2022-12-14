import numpy as np
import pandas as pd
from pathlib import Path
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

def pi_remove_offset(array3_raw, fs, size):
    """
    removes pressure offset during swing
    :param array3_raw: 3D array with pressure data from 1 side (31 x 11 x frames)
    :param fs: sampling rate - int
    :return: offset removed pressure data
    """
    # calculate force
    force = pi_force(array3_raw, size)
    # calculate IC
    ic = pi_ic(force, fs)

    array3 = np.empty(array3_raw.shape)
    # for every ic: find largest offset frame in previous 0.25 s
    # substract this offset from all frames between current and previous IC
    for i in range(len(ic)):
        if i == 0:
            if ic[i]==0:
                continue
            else:
                array3[:, :, :ic[i]] = offset_stage1(array3_raw[:, :, :ic[i]], fs)
        else: # maybe remove "-1"
            array3[:, :, ic[i-1]:ic[i]] = offset_stage1(array3_raw[:, :, ic[i-1]:ic[i]], fs)

    return array3

def pi_force(array3, size):
    return np.sum(np.sum(array3, axis=0), axis=0) * size

def pi_ic(force_side, sampling_rate):

    # calculate rate of force development (1st derivative of force)
    rfd = np.diff(force_side, n=1)*sampling_rate   # unit N/s

    # find where rfd > 1500 N/s based on Seiberl et al. (2018)
    rfd_1500 = np.where(rfd > 1500)[0]
    rfd_1500 = list(rfd_1500[rfd_1500<len(force_side)-sampling_rate//20])
    IC_side = []

    # loop over all indices of rfd_1500
    for IC in rfd_1500:
        # check if force increases monotonically by at least 1000 N in next 0.05s
        # TODO: needs to be checked
        if ((force_side[IC+sampling_rate//20]) > (force_side[IC]+1000)) & (min(force_side[IC:IC+sampling_rate//20]) == force_side[IC:IC+sampling_rate//20][0]):
            # first ic
            if len(IC_side) == 0:
                IC_side.append(IC)
            # all further ic have to be at least 0.25s apart
            elif IC > IC_side[-1]+sampling_rate//4:
                IC_side.append(IC)

    return np.array(IC_side, dtype=int)

def find_max_offset(array3, fs):
    if array3.shape[2] < fs//4:
        offset = np.mean(array3, axis=2)
        # offset = array3[:, :, 0]
        # for frame in range(array3.shape[2]):
        #     if np.sum(offset)<np.sum(array3[:,:,frame]):
        #         offset = array3[:,:,frame]
    else:
        offset = np.mean(array3[:, :, -fs//4:], axis=2)
        # for frame in range(-fs//4,-1):
        #     if np.sum(offset)<np.sum(array3[:,:,frame]):
        #         offset = array3[:,:,frame]
    return offset

def offset_stage1(array3, fs):
    offset = find_max_offset(array3, fs)
    for frame in range(array3.shape[2]):
        array3[:,:,frame]-=offset
    array3[array3<0]=0
    return array3

def pi_step_segmentation(force_left, force_right, sampling_rate_pi):
    """
    :param force_left:
    :param force_right:
    :param sampling_rate_pi:
    :return:
    """
    IC_left = pi_ic(force_left, sampling_rate_pi)
    IC_right = pi_ic(force_right, sampling_rate_pi)
    TO_left = pi_to(force_left, IC_left, sampling_rate_pi)
    TO_right = pi_to(force_right, IC_right, sampling_rate_pi)
    events_dict = {'IC_left': IC_left[:-1],
              'IC_right': IC_right[:-1],
              'TO_left': TO_left,
              'TO_right': TO_right
    }
    return events_dict

def pi_to(force_side, IC_side, fs):
    # filter force data (below 50 N threshold --> 0)
    force_side[force_side < 50] = 0
    TO_side=[]

    # loop to find Toe Off after respective IC
    # force = 0 for the first time 0.1 s after IC
    # clip before last IC (there might not be a toe off after)
    for IC in IC_side[:-1]:
        pot_to = np.argwhere(force_side[IC+fs//10:] == 0)[0] + IC+fs//10
        TO_side.extend(pot_to)
    return np.array(TO_side, dtype=int)

def find_hop(events_dict):
    """
    finds hop (2 contacts of the same side) for syncing PI
    :param events_dict: dict with indices for IC & TO events
    :return: string 'IC_left'/'IC_right', index of first Hop in that array
    """
    if events_dict['IC_left'][0]<events_dict['IC_right'][0]:
        first = 'IC_left'
        second = 'IC_right'
    else:
        first = 'IC_right'
        second = 'IC_left'
    for i in range(20):
        if events_dict[first][i+1] < events_dict[second][i]:
            print('found hop at ' + str(i) + '. step '+ first[3:] + ' - index:' + str(events_dict[first][i]))
            return first, i
        elif events_dict[second][i] < events_dict[first][i]:
            print('found hop at ' + str(i) + '. step '+ second[3:] + ' - index: ' + str(events_dict[second][i]))
            return second, i

    print('No hop found in first 20 steps..')
    return

def pi_temporal_parameters(pi_events_dict, sampling_rate_pi):
    """
    calculates Ground Contact Time, Flight Time, Stride Time
    :param pi_events_dict: Dict containing all IC and TO events
    :param sampling_rate_pi: sampling rate fs
    :return temp_params_df: Multiindex df with GCT, Flight T., Stride T., Side for all Steps
    """

    st_left = np.diff(pi_events_dict['IC_left'] / sampling_rate_pi, n=1)
    st_right = np.diff(pi_events_dict['IC_right'] / sampling_rate_pi, n=1)
    gct_left = (pi_events_dict['TO_left'] - pi_events_dict['IC_left']) / sampling_rate_pi
    gct_right = (pi_events_dict['TO_right'] - pi_events_dict['IC_right']) / sampling_rate_pi

    IC_all = np.sort(np.concatenate([pi_events_dict['IC_right'], pi_events_dict['IC_left']]))[1:]
    TO_all = np.sort(np.concatenate([pi_events_dict['TO_right'], pi_events_dict['TO_left']]))[:-1]
    ft = (IC_all - TO_all) / 100
    if pi_events_dict['IC_right'][0] < pi_events_dict['IC_left'][0]:
        ft_right_off = (IC_all - TO_all)[::2] / sampling_rate_pi
        ft_left_off = (IC_all - TO_all)[1::2] / sampling_rate_pi
    else:
        ft_left_off = (IC_all - TO_all)[::2] / sampling_rate_pi
        ft_right_off = (IC_all - TO_all)[1::2] / sampling_rate_pi

    cols = [np.repeat(['Stride Time [s]', 'Stance Time [s]', 'Flight Time [s]'], 2), np.tile(['left', 'right'], 3)]

    temp_params_df = pd.DataFrame([st_left, st_right, gct_left, gct_right, ft_left_off, ft_right_off], index=cols).T
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
        p = np.empty(pi_steps[key].shape[2])

        # calculate parameters for each frame
        for frame in range(pi_steps[key].shape[2]):
            # Area: count all non-zero sensors, multiply by sensor size
            a[frame] = np.count_nonzero(pi_steps[key][:,:,frame])*sensor_size
            # Force: sum up all pressure values
            f[frame] = np.sum(pi_steps[key][:,:,frame])
            # pressure = F/A when A=!0
            if a[frame] == 0:
                p[frame] = 0
                f[frame] = 0
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

def get_size_units(filepath):
    filepath = Path(filepath)
    current = filepath.name
    raw_dir = filepath.parent.name.replace('df', 'raw')
    raw_path = filepath.parents[1] / raw_dir / current

    # load raw
    df = pd.read_csv(raw_path, sep=',', header=None, nrows=22)
    p_unit = df[df[0]=='Units:'].iloc[0,1]
    size = float(df[df[0]=='Sensel Width (cm)'].iloc[0,1].replace(',','.')) * \
           float(df[df[0]=='Sensel Height (cm)'].iloc[0,1].replace(',','.'))

    return size, p_unit
