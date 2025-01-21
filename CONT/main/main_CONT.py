# import required packages
import os
from pathlib import Path
import configparser
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import matplotlib.pyplot as plt

# import dataloader
from CONT.utils.data import create_dataloader
from CONT.utils.data import PiImuToMomentsDataset as DATA_SET

# import model
from CONT.utils.model import PiImutoMomentsNet as MODEL_CLASS

# import train eval routine
from CONT.utils.training import TrainingRoutine as TRAINING

# import custom loss
from CONT.utils.loss import AxisWiseLoss_RMSE as CRITERION

# plot predicitons
from CONT.utils.plot import plot_prediction as PLOT
from CONT.utils.plot import plot_loss

# calculate RMSE & nRMSE
from CONT.utils.utils import model_statistics

# go to root directory
os.chdir("..")

# get device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# load config
config_name = 'config/CONT_config.txt'
cfg = configparser.ConfigParser()
cfg.read(config_name)
# print('\n'.join(f"[{section}]\n    {option} = {value}" if option == next(iter(cfg[section])) else f"    {option} =
# {value}" for section in cfg.sections() for option, value in cfg[section].items()))

# set the initial value for pseudo-random functions
np.random.seed(cfg.getint('experiment', 'seed'))
torch.manual_seed(cfg.getint('experiment', 'seed'))

# Load Variables
PATH_DATA = Path(cfg['settings']['data_path'])
TENSOR_LENGTH = cfg.getint('training_parameters', 'tensor_length')
MODEL = MODEL_CLASS(cfg)
VAL_SUBJ = [cfg.getint("experiment", "val_subj")]

# Loss Variables
SX = cfg.getfloat('loss_parameters', 'scale_x')  # frontal plane
SY = cfg.getfloat('loss_parameters', 'scale_y')  # transverse plane
SZ = cfg.getfloat('loss_parameters', 'scale_z')  # sagittal plane
criterion = CRITERION(weights=(SX, SY, SZ))

# create data loaders (participant wise split)
train_loader, val_loader = create_dataloader(cfg, DATA_SET)

# train the model
training = TRAINING(
    device=device,
    cfg=cfg,
    model=MODEL,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader
)

EPOCHS = cfg.getint('experiment', 'n_epochs')
for epoch in tqdm(range(EPOCHS)):
    training.train(epoch=epoch, verbose=0)
    training.val(epoch=epoch, verbose=0)
    print(f"Epoch {epoch + 1}/{EPOCHS}: \ttrain: {training.train_losses[epoch]}, \teval: {training.val_losses[epoch]}")

# calculate mean RMSE & nRMSE of validation set
mean_rmse, mean_nrmse = model_statistics(cfg, MODEL, device, val_loader)
print(f'RMSE is {mean_rmse.detach().numpy()} and nRMSE is {mean_nrmse.detach().numpy()}'
      f' for frontal, transverse & sagittal plane, respectively.')

# get loss over epochs
loss_df = pd.DataFrame({
    "epoch": np.arange(1, EPOCHS + 1),
    "train": training.train_losses,
    "val": training.val_losses})

# plot loss
plot_loss(loss_df)

# plot prediction examples
PLOT(MODEL, train_loader, device, loader="train")
PLOT(MODEL, val_loader, device, loader="val")
plt.show(block=True)
