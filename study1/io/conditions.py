import glob2
import pandas as pd
import numpy as np

def imu_path(participant_path):
    # get all txt files = all imu files
    return glob2.glob(participant_path + '\\*.txt')

def pi_path(participant_path):
    # get all csv files = all pi files
    return glob2.glob(participant_path + '\\*.csv')

def get_conditions(participant_path):
    imu_files = imu_path(participant_path)
    imu_files.sort()
    pi_files = pi_path(participant_path)
    pi_files.sort()
    participant = []
    shoe = []
    slope = []
    surface = []
    speed = []
    imu = []
    pi = []

    if len(imu_files)<len(pi_files):
        for file in pi_files:
            if file[:-4] + '.txt' in imu_files:
                imu.append(True)
            else:
                imu.append(False)
            file = file.split('\\')
            cond = file[-1].split('_')
            participant.append(cond[0])
            shoe.append(cond[1])
            slope.append(cond[2])
            surface.append(cond[3])
            speed.append(cond[4][:-4])
            pi.append(True)
    else:
        for file in imu_files:
            if file[:-4]+'.csv' in pi_files:
                pi.append(True)
            else:
                pi.append(False)
            file=file.split('\\')
            cond = file[-1].split('_')
            participant.append(cond[0])
            shoe.append(cond[1])
            slope.append(cond[2])
            surface.append(cond[3])
            speed.append(cond[4][:-4])
            imu.append(True)


    conditions=pd.DataFrame([participant, shoe, slope, surface, speed, imu, pi]).T
    conditions.columns=['participant', 'shoe', 'slope', 'surface', 'speed', 'imu_data', 'pi_data']
    conditions['imu_path'] = 'no file'
    conditions['pi_path'] = 'no file'
    # TODO: this has to be adjusted at some point
    conditions['pi_fs'] = 'no file'
    conditions.loc[conditions['pi_data']==True ,'pi_path']=pi_files
    conditions.loc[conditions['pi_data'] == True, 'pi_fs'] = [150, 75, 150, 75, 150]
    conditions.loc[conditions['pi_data'] == True, 'pi_path'] = pi_files
    conditions.loc[conditions['imu_data'] == True, 'imu_path'] = imu_files

    return (conditions)

def load_pi(trial):
    pi_data = pd.read_csv(trial, index_col=0)
    pi_data.set_index(np.linspace(0, len(pi_data)-1, len(pi_data)))
    return pi_data

def load_imu(trial):
    return pd.read_csv(trial )

