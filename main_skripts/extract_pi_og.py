import pandas as pd
from pathlib import Path
import numpy as np
from study1.io.conditions import pi_path, load_pi
import study1.utilities.pi as pi
import matplotlib.pyplot as plt

participant_path = 'D:\\Salzburg\\Study1\\P02\\P02_PI_df'
conditions = pi_path(participant_path)
conditions.sort()
fs = 100

for trial in conditions:
    #trial =
    current = Path(trial).name[:-4]
    print(current)

    if current.split('_')[-2] == 'OG':
        size, p_unit = pi.get_size_units(trial)
        # load pi data
        df = load_pi(trial)
        # separate side
        left_raw, right_raw = pi.separate_sides(df)

        # convert units to N/cm²
        if p_unit == 'PSI':
            left_raw *= 0.6894757
            right_raw *= 0.6894757

        # calculate vertical force
        force_left = pi.pi_force(left_raw, size)
        force_right = pi.pi_force(right_raw, size)

        # plot
        plt.plot(force_left, label='Left')
        plt.plot(force_right, label='Right')
        plt.hlines(20,0,len(force_right))
        plt.title(current)
        plt.legend()
        plt.show()




