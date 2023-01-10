from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from study1.io.nexus_output import read_nexus_overground_csv
from study1.utilities.force import segments_steps
from study1.io.conditions import pi_path, load_pi
import study1.utilities.pi as pi
from study1.utilities.signal import downsample

### create reader df
df = pd.read_csv(r'C:\Users\b1090197\OneDrive\Documents\TexSense\Code\Study1\files\pi_og_idxs.csv', sep = ';')
df['File']=df['Participant'] + '_' + df['Shoe'] + '_OG_' + df['OG'].astype(str)
df['Date'] = None
for participant in df['Participant'].unique():
    p = Path('D:\Salzburg\Study1')/participant/(participant+'_OG')
    date = sorted(p.glob('*.csv'))[0].name.split('_')[0]
    df.loc[df['Participant']==participant, 'Date'] = date
df['path_nexus']=Path('D:/Salzburg/Study1/')/df['Participant']/(df['Participant']+'_OG')/(df['Date']+'_'+ df['File'] + '.csv')
df['path_pi']=Path('D:/Salzburg/Study1/')/df['Participant']/(df['Participant']+'_PI_df')/(df['File'] + '.csv')

# example
example=25
for example in range(170,len(df)):
    trial_nexus = df.loc[example, 'path_nexus']
    trial_pi = df.loc[example, 'path_pi']
    fs_pi = 100     # Hz
    print(trial_pi.name)
    ### load force data
    nexus_dict = read_nexus_overground_csv(trial_nexus)
    # downsample
    force_df = downsample(nexus_dict['df_force'], nexus_dict['fs_force'], fs_pi)
    ic, to = segments_steps(force_df, fs_pi)

    ### load pi data
    if df.loc[example,'L/R']=='L':
        pi_data, _ = pi.separate_sides(load_pi(trial_pi))
    else:
        _, pi_data = pi.separate_sides(load_pi(trial_pi))
    size, p_unit = pi.get_size_units(trial_pi)
    # convert units to N/cm²
    if p_unit == 'PSI':
        pi_data *= 0.6894757
    # calculate force
    force_pi = pi.pi_force(pi_data[:,:,df.loc[example,'w1']-20:df.loc[example,'w2']+20], size)
    ic_pi = pi.pi_ic_test(force_pi, fs_pi)
    delay = ic_pi - ic[0]
    if delay < 0:
        print('IC pi before IC.. next')
        continue

    ###
    plt.plot(force_pi[delay:])
    plt.plot(ic_pi-delay, force_pi[delay:][ic_pi-delay], 'x')
    plt.plot(-force_df['Fz'])
    plt.plot(ic, -force_df['Fz'][ic], 'x')
    plt.plot(to, -force_df['Fz'][to], 'o')

    #plt.legend()
    plt.show()


