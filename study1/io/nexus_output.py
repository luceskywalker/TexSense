import pandas as pd
import numpy as np

def read_nexus_csv(filepath):
    """
    reads nexus csv file and imports IMU, Force and Marker Data
    returns dict with Data in DF (float) and Sampling Frequencies (int)
    looks like this:
            'df_imu': regular df
            'fs_imu': int
            'df_force': regular df
            'fs_force': fs_force,
            'df_marker': df_marker,
            'fs_marker': fs_marker
    :param filepath:
    :return following dict:
            'df_imu': regular df
            'fs_imu': int
            'df_force': regular df
            'fs_force': int,
            'df_marker': Multiindex DF (1st level: Marker names, 2nd level ['x', 'y', 'z']),
            'fs_marker': int
    """
    # read file in 1 columns with strings
    df = pd.read_csv(filepath, sep=';', header=None)

    # get indices of Device Data
    idx_imu = df[df[0] == 'Devices'].index[0]
    idx_force = df[df[0] == 'Devices'].index[1]
    idx_marker = df[df[0] == 'Trajectories'].index[0]

    # slice string df
    df_imu = df[idx_imu:idx_force]
    df_force = df[idx_force:idx_marker]
    df_marker = df[idx_marker:]

    # get sampling rates in Hz
    fs_imu = int(df_imu.iloc[1])
    fs_force = int(df_force.iloc[1])
    fs_marker = int(df_marker.iloc[1])

    # reshape Device Data and store in DF
    df_imu = df_reshape_imu(df_imu)
    df_force = df_reshape_force(df_force)
    df_marker = df_reshape_marker(df_marker)
    # divide by 1000 to convert acc data from mm/s² to m/s²
    df_imu.iloc[:, :len(df_imu.columns) // 2] = df_imu.iloc[:, :len(df_imu.columns) // 2] / 1000

    # save in dict
    nexus_dict = {
        'df_imu': df_imu,
        'fs_imu': fs_imu,
        'df_force': df_force,
        'fs_force': fs_force,
        'df_marker': df_marker,
        'fs_marker': fs_marker
    }

    return nexus_dict

def df_reshape_imu(df):
    cols = df.iloc[3].values[0].split(',')[2:]
    df = df[0].str.split(',', expand=True).iloc[5:, 2:-1]
    df.columns = cols
    df.reset_index(drop=True, inplace=True)
    return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

def df_reshape_force(df):
    cols = df.iloc[3].values[0].split(',')[6:]
    df = df[0].str.split(',', expand=True).iloc[5:, 6:-1]
    df.columns = cols
    df.reset_index(drop=True, inplace=True)
    return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

def df_reshape_marker(df_in):
    markers = [name[4:] for name in df_in.iloc[2].values[0].replace(',,', '').split(',')[:-1]]
    cols = [np.repeat(markers, 3), np.tile(['x', 'y', 'z'], len(markers))]
    df_out = df_in[0].str.split(',',expand=True).iloc[5:,2:-1]
    df_out.columns=cols
    df_out.reset_index(drop=True, inplace=True)
    return df_out.apply(lambda x: pd.to_numeric(x, errors='coerce'))

def get_force_imu_data(filepath, fs_imu=2000):
    """
    load force data (computed by nexus), imu data and extract sampling rates
    :param filepath: current csv file path
    :param fs_imu: int with IMU sampling rate, default = 2000 Hz
    :return force_df: df with force data
    :return fs_force: int with force sampling rate
    :return imu_df: df with force data
    :return fs_imu: int with force sampling rate
    """
    df = pd.read_csv(filepath, sep=',', header=[3], low_memory=False)
    # get force data
    ff = df.iloc[df[df['Frame']=='Frame'].index[0]+2:, :]
    ff.columns = df.iloc[df[df['Frame'] == 'Frame'].index[0], :].values.tolist()
    force_df = ff[['Frame', 'Sub Frame', 'Fx','Fy','Fz','Mx','My','Mz','Cx','Cy','Cz']].astype('float')
    force_df.reset_index(inplace=False)
    # get fs_force
    fs_force = int(len(force_df)/(df[df['Frame']=='Frame'].index[0]-4)*fs_imu) # number of frames force/number of frames IMU * fs IMU (2000 Hz)

    # get imu data
    imu_df = df.iloc[1:df[df['Frame'] == 'Devices'].index[0], :].astype('float')
    if fs_force != fs_imu:
        imu_data = imu_df[::fs_imu//fs_force]
    imu_df.reset_index(inplace=False)

    return force_df, fs_force, imu_df, fs_imu



