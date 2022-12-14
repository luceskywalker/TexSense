import matplotlib.pyplot as plt

def plot_force_OG(trial, force_df, ic, to, save=True):
    # docstring
    # reverse Forces
    force_df *= -1
    current = trial.split('\\')[-1][13:-4]

    fig = plt.figure(figsize=(22, 10))
    for col in force_df.columns:
        plt.plot(force_df[col], label = col, linewidth=3)
    plt.plot(ic, force_df['Fz'][ic], 'rX', markersize=15, label = 'initial contact')
    plt.plot(to, force_df['Fz'][to], 'ro', markersize = 10, label = 'toe off')
    plt.rc('font', size=20)  # fontsize of the axes title
    plt.legend()
    plt.ylabel('Force [N]')
    plt.xlabel('Frames')
    plt.xlim(0,800)

    fig.suptitle(current, fontsize=30, y=.92)
    if save == True:
        plt.savefig(trial[:-4] + '_GRF.jpg', bbox_inches='tight')
    plt.close(fig)

    return

