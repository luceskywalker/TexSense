import pandas as pd


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
    imu_data = df.iloc[1:df[df['Frame'] == 'Devices'].index[0], :].astype('float')
    if fs_force != fs_imu:
        imu_data = imu_data[::fs_imu//fs_force]
    imu_data.reset_index(inplace=False)

    return force_df, fs_force, imu_data, fs_imu



