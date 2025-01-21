# import required packages
import os
import torch
import matplotlib.pyplot as plt

# import dataloader
from CONT.utils.data import create_dataloader
from CONT.utils.data import PiImuToMomentsDataset as DATA_SET

# import model
from CONT.utils.model import PiImutoMomentsNet as MODEL_CLASS

# plot predicitons
from CONT.utils.plot import plot_prediction as PLOT

# calculate RMSE & nRMSE
from CONT.utils.utils import model_statistics

# go to root directory
os.chdir("..")

# get device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# load pretrained model_dict and unpack cfg and STATE_DICT
model_name = "foot_shank_thigh_pelvis"  # available: foot, foot_shank, foot_shank_pelvis, foot_shank_thigh_pelvis
model_dict = torch.load(f"models/{model_name}.pt", weights_only=False, map_location=device)
cfg = model_dict["cfg"]
STATE_DICT = model_dict["STATE_DICT"]

# init model with cfg and load state dict with pretrained weights and biases
MODEL = MODEL_CLASS(cfg)
MODEL.load_state_dict(STATE_DICT)

# create validation set e.g. subject 1
cfg.set("experiment", "val_subj", "1")

# create val_loader
_, val_loader = create_dataloader(cfg, DATA_SET)

# calculate mean RMSE & nRMSE of validation set
mean_rmse, mean_nrmse = model_statistics(cfg, MODEL, device, val_loader)
print(f'RMSE is {mean_rmse.detach().numpy()} and nRMSE is {mean_nrmse.detach().numpy()}'
      f' for frontal, transverse & sagittal plane, respectively.')

# plot prediction examples - could be a loop, too
PLOT(MODEL, val_loader, device, loader="val")
plt.show(block=True)
