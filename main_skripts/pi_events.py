import pandas as pd
import numpy as np
from study1.io.conditions import pi_path, load_pi
import study1.utilities.pi as pi

participant_path = 'D:\\Salzburg\\Study1\\P01\\P01_PI_df'
conditions = pi_path(participant_path)
fs = 100
size = 0.82**2

for trial in conditions:
    df = load_pi(trial)
    current = trial.split('\\')[-1][:-4]
    print(current)

    # separate side
    left_raw, right_raw = pi.separate_sides(df)
    # remove offset
    left = pi.pi_remove_offset(left_raw, fs)
    right = pi.pi_remove_offset(right_raw, fs)

    # force
    force_left = pi.pi_force(left)
    force_right = pi.pi_force(right)

    # TODO: add plots
    # events
    events_dict = pi.pi_step_segmentation(force_left, force_right, fs)
    print(pi.find_hop(events_dict))
    # temporal parameters
    temp_params_df = pi.pi_temporal_parameters(events_dict, fs)
    print(temp_params_df[temp_params_df.columns[2:4]])
    temp_params_df.to_csv(participant_path+'\\temp_params_' + current + '.csv', index=False)

    # separate steps into dict
    pi_steps = pi.pi_separate_steps(left, right, events_dict)

    # COP, Force, Pressure, Area
    cop_dict = pi.pi_get_cop(pi_steps)
    fpa_dict = pi.pi_get_fpa(pi_steps, size)
    fpa_df = pi.pi_dict_resample(fpa_dict, 100)
    fpa_df.to_csv(participant_path+'\\FPA_'+current+'.csv', index = False)
    print('... next')
