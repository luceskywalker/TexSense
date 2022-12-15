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

trial = 'D:\\Salzburg\\Study1\\P01\\P01_PI_df\\UB_up_8.csv'
current = trial.split('\\')[-1][:-4]
print(current)
size, p_unit = pi.get_size_units(trial)

# load pi data
df = load_pi(trial)

# load force data, downsample and calculate true events
trial_nexus = path_nexus + current + '.csv'
nexus_dict = read_nexus_treadmill_csv(trial_nexus)
force_df = downsample(nexus_dict['df_force'], nexus_dict['fs_force'], fs)
ic_true, to_true = segments_steps(force_df, fs)

# separate side
left_raw, right_raw = pi.separate_sides(df)
# convert units to N/cm²
# if p_unit == 'PSI':
#     left_raw *= 0.6894757
#     right_raw *= 0.6894757

# remove offset
left = pi.pi_remove_offset(left_raw, fs, size)
right = pi.pi_remove_offset(right_raw, fs, size)

# force
force_left = pi.pi_force(left, size)
force_right = pi.pi_force(right, size)

# events
events_dict = pi.pi_step_segmentation(force_left, force_right, fs)

# sync force to pi
key_hop, idx_hop = pi.find_hop(events_dict)
force_delay = events_dict[key_hop][idx_hop] - ic_true[16]

force=-force_df['Fz']
force.index = force.index + force_delay
ic_true = ic_true + force_delay
to_true = to_true + force_delay

# resync pi (other insole) to force
pi_delay = events_dict['IC_left'][3] - ic_true[14]
force_left = pd.Series(force_left)
force_left.index = force_left.index - pi_delay
events_dict['IC_left'] = events_dict['IC_left'] - pi_delay
events_dict['TO_left'] = events_dict['TO_left'] - pi_delay

# plot
piplot.force_events(trial, force, ic_true, to_true, force_left, force_right, events_dict, save=True)
#piplot.events(trial, force_left, force_right, events_dict)

print('... next')
