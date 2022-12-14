from study1.io.conditions import get_conditions
from study1.io.conditions import load_pi
from study1.io.conditions import load_imu
from study1.utilities.pi import separate_sides
from study1.utilities.pi import pi_force
from study1.utilities.pi import pi_step_segmentation
from study1.utilities.pi import pi_temporal_parameters
from study1.utilities.pi import pi_separate_steps
from study1.utilities.pi import pi_get_cop
from study1.utilities.pi import pi_get_fpa
from study1.utilities.pi import pi_dict_resample
from study1.plot import pi_plots

# 1. get conditions file for Participant incl path
participant_path = 'D:\\Salzburg\\TexSense\\Messungen\\IMU+PI\\data\\Neuer Ordner'
conditions = get_conditions(participant_path)

# 2.1 load trial data - PI
for i in range(len(conditions[conditions['pi_data']==True])):
    trial = conditions.loc[conditions['pi_data']==True ,'pi_path'].iloc[i]
    pi_data = load_pi(trial)

# 2.2 Process Data

# 2.2.1  separate sides & reshape into 3D Array [31 x 11 x number of frames]
    pi_left, pi_right = separate_sides(pi_data)

#   - (Filter)

#   - reshape (by frames)
#   - calculate force [N]
    force_left = pi_force(pi_left, size)
    force_right = pi_force(pi_right, size)

#   - (calculate rate of force development)
#   - step segmentation
    sampling_rate_pi = conditions.loc[conditions['pi_data']==True ,'pi_fs'].iloc[i]
    pi_events = pi_step_segmentation(force_left, force_right, sampling_rate_pi)

#   - calculate GCT, Stride Time, Stance Time
# TODO: This function does not yet work
    temp_params_df = pi_temporal_parameters(pi_events, sampling_rate_pi)

#   - extract steps (stance) from continuous data (sides)
    pi_steps = pi_separate_steps(pi_left, pi_right, pi_events)

#   - calculate COP trajectory
    cop_steps = pi_get_cop(pi_steps)

#   - segment area (forefoot, midfoot, rearfoot, med/lat)

#   - calculate TS parameters (Force, Pressure, Contact Area)
    sensor_size = .82**2
    fpa_steps = pi_get_fpa(pi_steps, sensor_size)

#   - resample
    output_length = 101
    fpa_df = pi_dict_resample(fpa_steps, output_length)

#   - (plot)
    #pi_plots.force_events(force_left, force_right, pi_events)
    #pi_plots.mean_fpa(fpa_df)
#
# 3.1 load trial data - IMU
for i in range(len(conditions[conditions['imu_data']==True])):
    trial = conditions.loc[conditions['imu_data']==True, 'imu_path'].iloc[i]
    imu_data = load_imu(trial)

# 3.2. separate sides
#   - step segmentation
#   - calculate GCT, Stride Time, Flight Time
#   - resample
#   - (plot)
