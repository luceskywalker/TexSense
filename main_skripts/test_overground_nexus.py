from study1.io.conditions import nexus_path
from study1.io.nexus_output import read_nexus_overground_csv
from study1.plot.force_plots import plot_force_OG
from study1.utilities.force import segments_steps

path_nexus = 'D:\\Salzburg\\Study1\\P01\\P01_OG\\20221205_P01_'
conditions = nexus_path(path_nexus)

for trial in conditions:
    #trial = 'D:\\Salzburg\\Study1\\P01\\P01_PI_df\\UB_up_8.csv'
    current = trial.split('\\')[-1][13:-4]
    print(current)
    if current.split('_')[-2] != 'OG':
        continue

    # load & parse data
    nexus_dict = read_nexus_overground_csv(trial)
    ic, to = segments_steps(nexus_dict['df_force'], nexus_dict['fs_force'])

    plot_force_OG(trial, nexus_dict['df_force'].iloc[:,:3], ic, to)

    # print header of each df
    # print(nexus_dict['df_imu'].head()[:2])
    # print(nexus_dict['df_force'].head()[:2])
    # print(nexus_dict['df_marker'].head()[:2])

    print('... next\n')


