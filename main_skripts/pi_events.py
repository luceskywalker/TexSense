import pandas as pd
import numpy as np
from study1.io.conditions import pi_path, load_pi
import study1.utilities.pi as pi
import study1.plot.pi_plots as piplot
from study1.io.nexus_output import read_nexus_treadmill_csv
from study1.utilities.signal import downsample
from study1.utilities.force import segments_steps

participant_path = 'D:\\Salzburg\\Study1\\P01\\P01_PI_df'
conditions = pi_path(participant_path)
fs = 100
size = 0.82**2
path_nexus = 'D:\\Salzburg\\Study1\\P01\\P01_treadmill\\20221205_P01_'
for trial in conditions:
    #trial = 'D:\\Salzburg\\Study1\\P01\\P01_PI_df\\UB_up_8.csv'
    current = trial.split('\\')[-1][:-4]
    print(current)

    if (current.split('_')[1] != 'OG') & (current.split('_')[0] == 'UB'):
        # load pi data
        df = load_pi(trial)

        # load force data, downsample and calculate true events
        trial_nexus = path_nexus + current + '.csv'
        nexus_dict = read_nexus_treadmill_csv(trial_nexus)
        force_df = downsample(nexus_dict['df_force'], nexus_dict['fs_force'], fs)
        ic_true, to_true = segments_steps(force_df, fs)

        # separate side
        left_raw, right_raw = pi.separate_sides(df)
        # remove offset
        left = pi.pi_remove_offset(left_raw, fs)
        right = pi.pi_remove_offset(right_raw, fs)

        # force
        force_left = pi.pi_force(left)
        force_right = pi.pi_force(right)

        # events
        events_dict = pi.pi_step_segmentation(force_left, force_right, fs)
        if current.split('_')[1] != 'OG':
            print(pi.find_hop(events_dict))
        # piplot.events(trial, force_left, force_right, events_dict, save=True)
        piplot.force_events(trial, -force_df["Fz"], ic_true, to_true, force_left, force_right, events_dict, save=True)

        # temporal parameters
        # temp_params_df = pi.pi_temporal_parameters(events_dict, fs)
        # print(temp_params_df[temp_params_df.columns[2:4]])
        # # temp_params_df.to_csv(participant_path+'\\output\\temp_params_' + current + '.csv', index=False)
        #
        # # separate steps into dict
        # pi_steps = pi.pi_separate_steps(left, right, events_dict)
        #
        # # COP, Force, Pressure, Area
        # #cop_dict = pi.pi_get_cop(pi_steps)
        # fpa_dict = pi.pi_get_fpa(pi_steps, size)
        # fpa_df = pi.pi_dict_resample(fpa_dict, 100)
        # fpa_df.to_csv(participant_path+'\\output\\FPA_'+current+'.csv', index = False)
        print('... next')
