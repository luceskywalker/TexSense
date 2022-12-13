import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#sns.set()
def events(trial, force_left, force_right, pi_events, save = True):
    """

    :param current: string with current trial
    :param force_left: Normal force over whole trial
    :param force_right: Normal force over whole trial
    :param pi_events: dict with all IC & TO Indices
    :param save: Bool for saving (default = True)
    :return:
    """
    current = trial.split('\\')[-1][:-4]
    fig = plt.figure(figsize=(32, 14))
    plt.plot(force_left, label='left')
    plt.plot(force_right, label='right')
    plt.plot(pi_events['IC_right'], force_right[pi_events['IC_right']], 'rx')
    plt.plot(pi_events['TO_right'], force_right[pi_events['TO_right']], 'ro')
    plt.plot(pi_events['IC_left'], force_left[pi_events['IC_left']], 'bx')
    plt.plot(pi_events['TO_left'], force_left[pi_events['TO_left']], 'bo')
    plt.rc('font', size=16)  # fontsize of the axes title
    plt.legend()
    plt.ylabel('Normal Force [N]')
    plt.xlabel('Time [ms]')
    # plt.tight_layout()
    fig.suptitle(current, fontsize=30, y=.92)
    if save == True:
        if current.split('_')[1] != 'OG':
            plt.xlim(0, 2000)
            plt.savefig(trial[:-4] + '-1.jpg', bbox_inches='tight', dpi = 60)
            plt.xlim(2000, 4000)
            plt.savefig(trial[:-4] + '-2.jpg', bbox_inches='tight', dpi = 60)
            plt.xlim(4000, 6000)
            plt.savefig(trial[:-4] + '-3.jpg', bbox_inches='tight', dpi = 60)
        else:
            plt.savefig(trial[:-4] + '.jpg', bbox_inches='tight', dpi = 60)
    plt.close(fig)

    return

def force_events(trial, force, ic, to, force_left, force_right, pi_events, save = True):
    """

    :param current: string with current trial
    :param force_left: Normal force over whole trial
    :param force_right: Normal force over whole trial
    :param pi_events: dict with all IC & TO Indices
    :param save: Bool for saving (default = True)
    :return:
    """
    current = trial.split('\\')[-1][:-4]
    fig = plt.figure(figsize=(32, 14))

    plt.plot(force_left, label='left', linewidth=3)
    plt.plot(force_right, label='right', linewidth=3)
    plt.plot(pi_events['IC_right'], force_right[pi_events['IC_right']], 'rX', markersize=15)
    plt.plot(pi_events['TO_right'], force_right[pi_events['TO_right']], 'ro', markersize=10)
    plt.plot(pi_events['IC_left'], force_left[pi_events['IC_left']], 'bX', markersize=15)
    plt.plot(pi_events['TO_left'], force_left[pi_events['TO_left']], 'bo', markersize=10)

    plt.plot(force, 'k--', label = 'Fvert', linewidth=2)
    plt.plot(ic, force[ic], 'kX', markersize=15)
    plt.plot(to, force[to], 'ko', markersize=10)

    plt.rc('font', size=20)  # fontsize of the axes title
    plt.legend()
    plt.ylabel('Force [N]')
    plt.xlabel('Time [ms]')
    # plt.tight_layout()
    fig.suptitle(current, fontsize=30, y=.92)
    if save == True:
        if current.split('_')[1] != 'OG':
            plt.xlim(0, 750)
            plt.savefig(trial[:-4] + '_force-1.jpg', bbox_inches='tight') #, dpi = 60)
            #plt.xlim(2000, 4000)
            #plt.savefig(trial[:-4] + '_force-2.jpg', bbox_inches='tight', dpi = 60)
            #plt.xlim(4000, 6000)
            #plt.savefig(trial[:-4] + '_force-3.jpg', bbox_inches='tight', dpi = 60)
        else:
            plt.savefig(trial[:-4] + '_force.jpg', bbox_inches='tight', dpi = 60)
    plt.close(fig)

    return

def mean_fpa(fpa_df):
    """
    plots mean Force, Pressure, Contact Area over all steps
    :param fpa_df: Multi-index DF with resampled Force, Pressure and Area data
    :return:
    """
    f = fpa_df['Force']
    f_left = f[f.columns[f.columns.str.contains(pat='left')]]
    f_right = f[f.columns[f.columns.str.contains(pat='right')]]
    a = fpa_df['Area']
    a_left = a[f.columns[a.columns.str.contains(pat='left')]]
    a_right = a[a.columns[a.columns.str.contains(pat='right')]]
    p = fpa_df['Pressure']
    p_left = p[p.columns[p.columns.str.contains(pat='left')]]
    p_right = p[p.columns[p.columns.str.contains(pat='right')]]

    fig, axs = plt.subplots(1,3, constrained_layout = True)
    x = len(fpa_df)
    axs[0].fill_between(x, np.mean(f_left, axis = 1)+np.std(f_left, axis = 1),
                        np.mean(f_left, axis = 1)-np.std(f_left, axis = 1), label='Left')
    axs[0].fill_between(x, np.mean(f_right, axis = 1)+np.std(f_right, axis = 1),
                        np.mean(f_right, axis = 1)-np.std(f_right, axis = 1), label='Right')
    axs[1].fill_between(x, np.mean(p_left, axis=1) + np.std(p_left, axis=1),
                        np.mean(p_left, axis=1) - np.std(p_left, axis=1), label='Left')
    axs[1].fill_between(x, np.mean(p_right, axis=1) + np.std(p_right, axis=1),
                        np.mean(p_right, axis=1) - np.std(p_right, axis=1), label='Right')
    axs[2].fill_between(x, np.mean(a_left, axis=1) + np.std(a_left, axis=1),
                        np.mean(a_left, axis=1) - np.std(a_left, axis=1), label='Left')
    axs[2].fill_between(x, np.mean(a_right, axis=1) + np.std(a_right, axis=1),
                        np.mean(a_right, axis=1) - np.std(a_right, axis=1), label='Right')
    plt.show()

    return

# ### Animation
# fig, ax = plt.subplots()
# ims=[]
# for i in range(stop_right[11]-start_right[11]):
#     im=ax.imshow(right_data[:,:,start_right[11]+i], animated=True, cmap='jet', interpolation='nearest')
#     if i==0:
#         ax.imshow(right_data[:,:,start_right[11]], cmap='jet', interpolation='nearest')
#     ims.append([im])
#
# ani=animation.ArtistAnimation(fig, ims, interval=26.6, blit=True, repeat_delay=10)
# plt.show()
#
# f = r"C:\Users\b1090197\Documents\Python\Untitled Folder\step_cycle.gif"
# writergif = animation.PillowWriter(fps=26.6)
# ani.save(f, writer=writergif)
#
# ### Plot Segment Lines
# plt.imshow(np.mean(right_mp, axis=2), cmap='jet', interpolation='nearest')
# plt.vlines(6, -0.5, 30.5, colors='r')
# plt.hlines([11,21], -0.5, 10.5, colors='r')
#
# ### Plots mean pressure in Segments
# for i in slices_right.keys():
#     plt.plot(np.mean(mean_pressure(slices_right[i]).values, axis=1) , label=i)
#     plt.legend()
#     plt.rcParams["figure.figsize"] = (8,6)
#     plt.xlim(0,100)
#     plt.ylim(0,22)
#     plt.title('mittlerer Druckverlauf der Segmente')
#     plt.xlabel('Stance [%]')
#     plt.ylabel('Pressure [N/cm²]')


