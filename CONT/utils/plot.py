# ToDo new
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

IDX = pd.IndexSlice


sns.set(style="whitegrid", context="notebook", rc={"figure.figsize": (8, 5)})

def plot_loss(df_results):
    """
    Plot training and validation loss from a DataFrame containing training results.

    Args:
        df_results (pandas.DataFrame): DataFrame containing training results with columns 'epoch', 'train', and 'val'.

    Returns:
        None
    """
    # Plot training and validation loss using Seaborn
    sns.lineplot(df_results, x="epoch", y="train", marker="o", label="train")
    sns.lineplot(df_results, x="epoch", y="val", marker="o", label="val")

    # Set plot title and axis labels
    plt.title("Training and Validation Error")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Set y-axis lower limit to 0
    plt.ylim(0, None)

    plt.show(block=False)

    return


def plot_prediction(model, data_loader, device, idx=None, loader=''):
    """
    Plot predicted moments and compare with ground truth, given an index of a data loader.

    Args:
        model (torch.nn.Module): The trained model for predicting moments.
        data_loader (torch.utils.data.DataLoader): Data loader containing the dataset.
        device (torch.device): Device on which to run the model (e.g., CPU or GPU).
        idx (int, optional): Index of the data loader to visualize. If None, a random index is chosen. Default is None.
        loader (str, optional): Identifier for the loader being used ('train', 'val', or 'test'). Default is an empty string.

    Returns:
        None
    """
    # predict
    model.eval()
    # If no index is provided, randomly choose one
    if idx is None:
        idx = np.random.randint(0, len(data_loader))

    # Extract sample data from the dataset
    sample = data_loader.dataset[idx]
    pi_data = sample["pi"]
    reference = sample['moments']
    imu = sample["imu"]
    aux = sample["level"]

    # Extract file name for display
    file = sample["file"].split(os.sep)[-1]

    # Perform prediction using the model
    pred = model(
        pi_data.reshape(1, pi_data.shape[0], -1).to(device),
        aux.reshape(1, aux.shape[0], -1).to(device),
        imu.reshape(1, imu.shape[0], -1).to(device)
    ).cpu().detach().numpy()

    # Prepare DataFrames for predicted and true moments
    df_pred = pd.DataFrame(pred.squeeze()[[2, 0, 1], :],
                           index=['sagittal (pred)', 'frontal (pred)', 'transverse (pred)']).T
    df_true = pd.DataFrame(reference.cpu().detach().numpy()[[2, 0, 1], :], index=['sagittal', 'frontal', 'transverse']).T

    # Convert index to seconds for easier interpretation
    df_pred.index = df_pred.index / 100
    df_true.index = df_true.index / 100

    # Plotting
    fig, ax = plt.subplots(3, figsize=[12, 8], sharex='all')
    fig.suptitle(f"Reference vs. Pred Knee Moments for '{'_'.join(file.split('_')[:5])}' ({loader.upper()})",
                 fontsize=18)
    for i, (color_pred, color_true) in enumerate(zip(['b', 'orange', 'g'], ['darkblue', 'darkorange', 'darkgreen'])):
        # Plot predicted and true moments for each direction
        df_pred.iloc[:, i].plot(ax=ax[i], color=color_pred, linewidth=4, alpha=.9)
        df_true.iloc[:, i].plot(ax=ax[i], color=color_true, linewidth=2.5)
        ax[i].set_title(df_true.columns[i])
        ax[i].set_xlim([1, 5])  # Limit x-axis to 1-5 seconds
        ax[i].set_xlabel('Time [s]')
        ax[i].legend(["predicted", "reference"], frameon=False, loc="upper right")
        ax[i].set_ylabel("[Nm/kg]")

    plt.tight_layout()
    plt.show(block=False)
