import pingouin as pg
import torch
import numpy as np
import pandas as pd


@torch.no_grad()
def get_model_performance(cfg, model, device, data_loader):
    """
       Compute performance metrics  statistics for the model predictions on the given data_loader.
       Performance metrics are RMSE, normalized RMSE, Pearson Correlation and ICC (Type 2)
       Metrics are computed over continuous samples (CONT) and during stance phases only (PHSS)

       Args:
           cfg (configparser.ConfigParser): Config object containing experiment settings.
           model (torch.nn.Module): Model used for predictions.
           device (torch.device): Device to perform computations on.
           data_loader (torch.utils.data.DataLoader): Data loader providing data for evaluation.

       Returns:
            2 dicts containing the results for CONT and PHSS
       """

    # Set model to evaluation mode
    model.eval()

    # init result dicts
    cont_dict = {}
    phss_dict = {}

    # Iterate through each sample in the data loader
    for idx in range(len(data_loader.dataset)):
        sample = data_loader.dataset[idx]
        if cfg.getboolean("sensor_setup", "pi"):
            pi_data = sample["pi"].to(device)
        else:
            pi_data = torch.tensor([1, 1]).to(device)
        true = sample["moments"].to(device)
        imu = sample["imu"].to(device)
        aux = sample['level'].to(device)
        name = sample["file"].split("\\")[1].split("_moment")[0]

        # Forward pass to get predictions
        pred = model(
            pi_data.reshape(1, pi_data.shape[0], -1).to(device),
            aux.reshape(1, aux.shape[0], -1).to(device),
            imu.reshape(1, imu.shape[0], -1).to(device)
        ).cpu().squeeze(0)  # [3, time]

        true = true.cpu()
        mask = true[2] != 0  # stance-phase mask

        # Compute per-channel metrics
        cont_rmse = torch_rmse(true, pred).tolist()
        cont_nrmse = torch_nrmse(true, pred).tolist()
        cont_r = torch_corr(true, pred).tolist()
        cont_icc = torch_icc(true, pred)

        # extract PHSS
        y_true_phss = true[:, mask]
        y_pred_phss = pred[:, mask]

        phss_rmse = torch_rmse(y_true_phss, y_pred_phss).tolist()
        phss_nrmse = torch_nrmse(y_true_phss, y_pred_phss).tolist()
        phss_r = torch_corr(y_true_phss, y_pred_phss).tolist()
        phss_icc = torch_icc(y_true_phss, y_pred_phss)

        cont_dict[name] = {
            "RMSE": cont_rmse,
            "nRMSE": cont_nrmse,
            "r": cont_r,
            "ICC": cont_icc
        }

        phss_dict[name] = {
            "RMSE": phss_rmse,
            "nRMSE": phss_nrmse,
            "r": phss_r,
            "ICC": phss_icc
        }

    return cont_dict, phss_dict

def torch_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=-1))


def torch_nrmse(y_true, y_pred):
    return torch_rmse(y_true, y_pred) / (y_true.max(dim=-1).values - y_true.min(dim=-1).values + 1e-8)


def torch_corr(y_true, y_pred):
    y_true_mean = y_true.mean(dim=-1, keepdim=True)
    y_pred_mean = y_pred.mean(dim=-1, keepdim=True)
    num = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=-1)
    den = torch.sqrt(torch.sum((y_true - y_true_mean) ** 2, dim=-1) * torch.sum((y_pred - y_pred_mean) ** 2, dim=-1))
    return num / (den + 1e-8)


def torch_icc(y_true, y_pred):
    """
    ICC(2,1) — Two-way random, absolute agreement, single rater.
    Vectorized torch version applied per channel.
    """
    n_ch, n_time = y_true.shape
    iccs = []

    for i in range(n_ch):
        x = torch.stack((y_true[i], y_pred[i]), dim=1)  # [n_time, 2]
        n, k = x.shape  # k=2

        mean_per_target = x.mean(dim=1, keepdim=True)
        mean_per_rater = x.mean(dim=0, keepdim=True)
        grand_mean = x.mean()

        MSR = torch.sum((mean_per_rater - grand_mean) ** 2) * n / (k - 1)
        MSC = torch.sum((mean_per_target - grand_mean) ** 2) * k / (n - 1)
        MSE = torch.sum((x - mean_per_target - mean_per_rater + grand_mean) ** 2) / ((k - 1) * (n - 1))

        ICC2 = (MSC - MSE) / (MSC + (k - 1) * MSE + k * (MSR - MSE) / n + 1e-8)
        iccs.append(ICC2.item())

    return iccs
