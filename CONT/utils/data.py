import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from CONT.utils.utils import center_of_mass_3d, foot_segmentation
import numpy as np
from configparser import ConfigParser


def create_dataloader(PATH_DATA: Path, VAL_SUBJ: list, cfg: ConfigParser, DATA_SET):
    """
    Function creating training and validation split and returning the corresponding data loaders.

    Args:
        PATH_DATA (Path): Path to the data directory.
        VAL_SUBJ (list): List of participants used for validation.
        cfg (ConfigParser): Configuration .
        DATA_SET: Torch dataset that should be used.

    Returns:
        tuple: Tuple containing train and validation data loaders.
    """
    # Create a list with all participant IDs as integers
    participant_ids = sorted(
        np.unique(
            [
                int(p_id.split("_")[0].replace("P", ""))
                for p_id in os.listdir(PATH_DATA)
                if p_id.endswith(".pt")
            ]
        )
    )

    # Make sure no validation participant is used in the training set
    train_ids = [x for x in participant_ids if x not in VAL_SUBJ]

    # Shuffle training participant IDs
    shuffled_ids = np.random.permutation(train_ids)

    # Print IDs for verification
    print(f"Train IDs: {shuffled_ids}")
    print(f"Val ID: {VAL_SUBJ}")

    # Create corresponding data loaders
    train_loader = DataLoader(
        DATA_SET(
            participant_ids=shuffled_ids,
            PATH_DATA=PATH_DATA,
            cfg=cfg,
        ),
        batch_size=cfg.getint("training_parameters", "batch_size"),
        shuffle=True,
    )
    val_loader = DataLoader(
        DATA_SET(
            participant_ids=VAL_SUBJ,
            PATH_DATA=PATH_DATA,
            cfg=cfg,
        ),
        batch_size=cfg.getint("training_parameters", "batch_size"),
        shuffle=True,
    )

    return train_loader, val_loader


class PiImuToMomentsDataset(Dataset):
    """Dataset class for moments, PI, and IMU data."""

    def __init__(
            self,
            participant_ids: list,
            PATH_DATA: Path,
            cfg: ConfigParser,
            transform=None,
    ):
        """
        Initialize the dataset with participant IDs, data paths, and configuration.

        Args:
            participant_ids (list): List of participant indices in the dataset.
            PATH_DATA (Path): Path to the data directory.
            cfg (ConfigParser): Configuration object containing model parameters.
            transform (function, optional): Transform function used for data augmentation.
        """
        self.cfg = cfg
        self.participant_ids = participant_ids
        self.path_moments = PATH_DATA
        self.path_pi = PATH_DATA
        self.transform = transform
        self.TENSOR_LENGTH = self.cfg.getint("training_parameters", "tensor_length")

        # Select all participant PI files
        file_list = [
            f for f in os.listdir(PATH_DATA) if f.startswith("P") and f.endswith(".pt")
        ]

        # Initialize lists to store file names for IMU, PI, and moments data
        self.file_list_imu = []
        self.file_list_pi = []
        self.file_list_moments = []

        # Select files that contain IMU and add other suffixes
        for file in file_list:
            if any(
                    file.startswith(f"P{str(p_id).zfill(2)}") and "_imu" in file
                    for p_id in self.participant_ids
            ):
                self.file_list_imu.append(file)
                pi_file = file.replace('imu', 'pi')
                moments_file = pi_file.replace('pi', 'moments')
                self.file_list_pi.append(pi_file)
                self.file_list_moments.append(moments_file)

    def __len__(self):
        """Returns the total number of files available."""
        return len(self.file_list_moments)

    def __getitem__(self, idx):
        """Returns a tuple (PI, moments, participant_id)."""
        # Load PI data
        tensor_pi = torch.load(os.path.join(self.path_pi, self.file_list_pi[idx]))
        tensor_pi = tensor_pi.reshape(-1, 31, 11)
        cop = center_of_mass_3d(tensor_pi)
        tensor_pi = torch.hstack([foot_segmentation(tensor_pi), cop])
        tensor_pi = tensor_pi.permute(1, 0)

        # Auxillary Information for Slope information
        if "_down_" in str(self.file_list_pi[idx]):
            level = torch.tensor([1, 0], dtype=torch.float32)
        elif "_level_" in str(self.file_list_pi[idx]):
            level = torch.tensor([0, 0], dtype=torch.float32)
        elif "_up_" in str(self.file_list_pi[idx]):
            level = torch.tensor([0, 1], dtype=torch.float32)
        else:
            raise Exception("No valid level found.")

        person_level = torch.tile(level, (self.TENSOR_LENGTH, 1))
        person_level = person_level.permute(1, 0)

        # Load IMU data
        tensor_imu = torch.load(os.path.join(self.path_pi, self.file_list_imu[idx]))
        tensor_imu = tensor_imu.permute(1, 0)

        # Load moments
        tensor_moments = torch.load(
            os.path.join(self.path_pi, self.file_list_moments[idx])
        )
        participant_id = int(self.file_list_moments[idx][1:].split("_")[0])
        tensor_moments = tensor_moments.permute(1, 0)

        # Zero-padding if the output length is not reached
        if tensor_pi.shape[1] != self.TENSOR_LENGTH:
            tensor_pi_full = torch.zeros(9, self.TENSOR_LENGTH)
            tensor_pi_full[:, : tensor_pi.size(1)] = tensor_pi
            tensor_pi = tensor_pi_full

        if tensor_imu.shape[1] != 4 * self.TENSOR_LENGTH:
            tensor_imu_full = torch.zeros(24, self.TENSOR_LENGTH * 4)
            tensor_imu_full[:, : tensor_imu.size(1)] = tensor_imu
            tensor_imu = tensor_imu_full

        if tensor_moments.shape[1] != self.TENSOR_LENGTH:
            tensor_moments_full = torch.zeros(3, self.TENSOR_LENGTH)
            tensor_moments_full[:, : tensor_moments.size(1)] = tensor_moments
            tensor_moments = tensor_moments_full

        sample = {
            "pi": tensor_pi,
            "moments": tensor_moments,
            "imu": tensor_imu,
            "participant_id": participant_id,
            "file": os.path.join(self.path_pi, self.file_list_moments[idx]),
            "level": person_level
        }

        return sample
