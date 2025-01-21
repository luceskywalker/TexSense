# import required packages
import os
import sys
from pathlib import Path
import configparser
import numpy as np
import torch
from sklearn.model_selection import KFold

# import optuna and logger for hyperopt
import optuna
import logging

# import dataloader
from CONT.utils.data import create_dataloader
from CONT.utils.data import PiImuToMomentsDataset as DATA_SET

# import model
from CONT.utils.model import PiImutoMomentsNet as MODEL_CLASS

# import train eval routine
from CONT.utils.training import TrainingRoutine as TRAINING

# import custom loss
from CONT.utils.loss import AxisWiseLoss_RMSE as CRITERION

# go to root directory
os.chdir("..")

# get device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# load config - make sure to specify the desired sensor setup
config_name = 'config/CONT_config.txt'
cfg = configparser.ConfigParser()
cfg.read(config_name)
# print('\n'.join(f"[{section}]\n    {option} = {value}" if option == next(iter(cfg[section])) else f"    {option} =
# {value}" for section in cfg.sections() for option, value in cfg[section].items()))

# define hypopt setting - number of iterations, number of folds for cross-validation, participants
N_TRIALS = 500
N_FOLDS = 3
ALL_USER_ID = [1, 2, 4, 5, 6, 7, 8, 9, 10]

# set logger
name_exp = f"{cfg.get('experiment', 'name')}_{cfg.get('experiment', 'suffix')}"
logger_name = f"hyperopt/log/{name_exp}.log"

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


def objective(trial):
    # set the initial value for pseudo-random functions
    np.random.seed(cfg.getint('experiment', 'seed'))
    torch.manual_seed(cfg.getint('experiment', 'seed'))

    # suggest parameters
    # [experiment]
    n_epochs = trial.suggest_int("n_epochs", 25, 50, step=5)
    merge_layer = trial.suggest_int("merge_layer", 0, 1, step=1)
    extra_layers = trial.suggest_int("extra_layers", 0, 2, step=1)

    # [training_parameters]
    batch_size_power = trial.suggest_int("batch_size_power", 3, 7, step=1)
    min_learning_rate = trial.suggest_float("min_learning_rate", 1e-4, 1e-3, log=True)
    learning_rate_ratio = trial.suggest_float("learning_rate_ratio", 1, 10, log=True)
    max_learning_rate = learning_rate_ratio * min_learning_rate

    # [model_parameters]
    # units/channels
    out_dim_pi_imu_power = trial.suggest_int("out_dim_pi_imu_power", 5, 8, step=1)
    hidden_dim_power = trial.suggest_int("hidden_dim_power", 5, 8, step=1)
    conv1_dim_power = trial.suggest_int("conv1_dim_power", 5, 8, step=1)
    if extra_layers >= 1:
        conv2_dim_power = trial.suggest_int("conv2_dim_power", 5, 8, step=1)
        if extra_layers == 2:
            conv3_dim_power = trial.suggest_int("conv3_dim_power", 5, 8, step=1)

    # kernel size
    kernel_in_idx = trial.suggest_int("kernel_in_idx", 0, 3, step=1)
    if merge_layer:
        kernel_merge_idx = trial.suggest_int("kernel_merge_idx", 0, 3, step=1)
    kernel_conv1_idx = trial.suggest_int("kernel_conv1_idx", 0, 3, step=1)
    if extra_layers >= 1:
        kernel_conv2_idx = trial.suggest_int("kernel_conv2_idx", 0, 3, step=1)
        if extra_layers == 2:
            kernel_conv3_idx = trial.suggest_int("kernel_conv3_idx", 0, 3, step=1)
    kernel_out_idx = trial.suggest_int("kernel_out_idx", 0, 3, step=1)

    # dropout & noise ratio
    # drop_ratio = trial.suggest_float("drop_ratio", 1e-3, 1, log=True)
    # noise_ratio = trial.suggest_float("noise_ratio", 1e-3, 1, log=True)

    # set parameters
    # [experiment]
    # cfg.set('experiment', 'dropout', str(bool(dropout)))
    cfg.set('experiment', 'n_epochs', str(n_epochs))
    cfg.set('experiment', 'merge_layer', str(bool(merge_layer)))
    cfg.set('experiment', 'extra_layers', str(extra_layers))

    # [training_parameters]
    cfg.set('training_parameters', 'batch_size', str(2 ** batch_size_power))
    cfg.set('training_parameters', 'min_learning_rate', str(min_learning_rate))
    cfg.set('training_parameters', 'max_learning_rate', str(max_learning_rate))

    # [model_parameters]
    # unit/channels: [16, 32, 64, 128]
    cfg.set('model_parameters', 'out_dim_pi_imu', str(2 ** out_dim_pi_imu_power))
    cfg.set('model_parameters', 'hidden_dim', str(2 ** hidden_dim_power))
    cfg.set('model_parameters', 'conv1_dim', str(2 ** conv1_dim_power))
    if extra_layers >= 1:
        cfg.set('model_parameters', 'x1_dim', str(2 ** conv2_dim_power))
        if extra_layers == 2:
            cfg.set('model_parameters', 'x2_dim', str(2 ** conv3_dim_power))

    kernel_sizes = [7, 15, 27, 51]
    cfg.set('model_parameters', 'kernel_in', str(kernel_sizes[kernel_in_idx]))
    if merge_layer:
        cfg.set('model_parameters', 'kernel_merge', str(kernel_sizes[kernel_merge_idx]))
    cfg.set('model_parameters', 'kernel_conv1', str(kernel_sizes[kernel_conv1_idx]))
    if extra_layers >= 1:
        cfg.set('model_parameters', 'kernel_x1', str(kernel_sizes[kernel_conv2_idx]))
        if extra_layers == 2:
            cfg.set('model_parameters', 'kernel_x2', str(kernel_sizes[kernel_conv3_idx]))
    cfg.set('model_parameters', 'kernel_out', str(kernel_sizes[kernel_out_idx]))

    # dropout & noise ratio
    # cfg.set('model_parameters', 'drop_ratio', str(drop_ratio))
    # cfg.set('model_parameters', 'noise_ratio', str(noise_ratio))

    # cross-validation
    mean_score = 0.0
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cfg.getint('experiment', 'seed'))
    for k_fold, (train_pids, val_pids) in enumerate(kf.split(ALL_USER_ID)):
        # set the validation subjects
        print(f"Fold '{k_fold}' with validation set '{val_pids}'")
        cfg.set('experiment', 'val_subj', "[" + ", ".join(
            [str(int(float(pid))) for pid in val_pids]
            ) + "]")

        # reinit the model with new parameters and train
        train_loader, val_loader = create_dataloader(cfg, DATA_SET)
        model = MODEL_CLASS(cfg)
        training = TRAINING(
            device=device,
            cfg=cfg,
            model=model,
            criterion=CRITERION(weights=(SX, SY, SZ)),
            train_loader=train_loader,
            val_loader=val_loader,
        )
        for epoch in range(n_epochs):
            training.train(epoch=epoch, verbose=0)
            training.val(epoch=epoch, verbose=0)

            # Report intermediate objective value.
            if k_fold == 0:
                trial.report(training.val_losses[-1], epoch)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # Score = VAL LOSS (AxisWiseRMSE)
        score = training.val_losses[-1]
        mean_score += score

    # empty cache
    torch.cuda.empty_cache()
    return mean_score / N_FOLDS


if __name__ == "__main__":
    # Create Optuna study
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    print(f"Starting Optimization with sampler {study.sampler.__class__.__name__}")

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(logger_name, mode="w"))

    study.optimize(
        objective, n_trials=N_TRIALS,
        gc_after_trial=True
        )
