import torch
import warnings

def center_of_mass_3d(tensor):
    """
    Calculates COP coordinates [x,y] relative to the center of a pressure image for each frame in the 3D input tensor.

    Args:
        tensor (torch.Tensor): 3D tensor representing pressure images [frames, 31, 11].

    Returns:
        torch.Tensor: 2D tensor representing COP coordinates [frames, 2].
    """
    # Set negative values in the tensor to 0
    tensor[tensor < 0] = 0

    # Create a grid of coordinates corresponding to each element in the tensor
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        grid = torch.meshgrid(torch.arange(tensor.shape[1]), torch.arange(tensor.shape[2]))
    grid = torch.stack(grid).to(tensor.device).float()  # Convert to float and move to the same device as the input tensor

    # Reshape the tensor to combine the batch dimension with the frame dimension
    reshaped_tensor = tensor.view(-1, tensor.shape[1], tensor.shape[2])

    # Calculate the sum of mass for each frame
    sum_mass = reshaped_tensor.sum(dim=(1, 2))

    # Calculate the center of mass for each frame
    cop_x = (reshaped_tensor * grid[1]).sum(dim=(1, 2)) / sum_mass
    cop_y = (reshaped_tensor * grid[0]).sum(dim=(1, 2)) / sum_mass

    # Transform the CoP coordinates to range from -1 to 1
    cop_x = (2 * (cop_x / (tensor.shape[2] - 1))) - 1
    cop_y = (2 * (cop_y / (tensor.shape[1] - 1))) - 1

    # Stack the x and y coordinates to get the final output tensor
    cop_tensor = torch.stack([cop_x, cop_y], dim=1)

    # Reshape the output tensor back to [frames, 2]
    cop_tensor = cop_tensor.view(tensor.shape[0], 2)

    # Adjust center of mass for frames with all-zero values to (0, 0)
    cop_tensor[sum_mass == 0] = torch.tensor([0.0, 0.0], device=cop_tensor.device)

    return cop_tensor


def foot_segmentation(array3):
    """
    Calculates the mean pressure for 7 foot segments out of a pressure image.

    Args:
        array3 (torch.Tensor): 3D tensor with PI data [frames, 31, 11].

    Returns:
        torch.Tensor: 2D tensor with mean pressure for 7 foot segments [frames, 7].
    """
    forefoot = array3[:, :11, :]
    midfoot = array3[:, 11:21, :]
    rearfoot = array3[:, 21:, :]

    def med_lat_slice(segment):
        med = segment[:, :, :6]
        lat = segment[:, :, 6:]
        return med, lat

    ffm, ffl = med_lat_slice(forefoot)
    mfm, mfl = med_lat_slice(midfoot)
    rfm, rfl = med_lat_slice(rearfoot)

    ffm_ant = ffm[:, :6, :]
    ffm_post = ffm[:, 6:, :]

    mean_tensor = torch.hstack([
        ffm_ant.mean(axis=[1, 2]).reshape(-1, 1),
        ffm_post.mean(axis=[1, 2]).reshape(-1, 1),
        ffl.mean(axis=[1, 2]).reshape(-1, 1),
        mfm.mean(axis=[1, 2]).reshape(-1, 1),
        mfl.mean(axis=[1, 2]).reshape(-1, 1),
        rfm.mean(axis=[1, 2]).reshape(-1, 1),
        rfl.mean(axis=[1, 2]).reshape(-1, 1),
    ])

    return mean_tensor


def model_statistics(cfg, model, device, data_loader):
    """
       Compute RMSE and normalized RMSE statistics for the model predictions on the given data_loader.

       Args:
           cfg (configparser.ConfigParser): Config object containing experiment settings.
           model (torch.nn.Module): Model used for predictions.
           device (torch.device): Device to perform computations on.
           data_loader (torch.utils.data.DataLoader): Data loader providing data for evaluation.

       Returns:
           tuple: A tuple containing mean RMSE and mean normalized RMSE over all samples and dimensions.
                  Each element of the tuple is a tensor with RMSE or normalized RMSE values for each dimension (sagittal, frontal, transversal).
       """
    # Set model to evaluation mode
    model.eval()
    # Lists to store RMSE and normalized RMSE values
    rmse_list, nrmse_list = [], []

    # Iterate through each sample in the data loader
    for idx in range(len(data_loader.dataset)):
        sample = data_loader.dataset[idx]
        pi_data = sample["pi"].to(device)
        true = sample["moments"]
        imu = sample["imu"].to(device)
        aux = sample['level'].to(device)

        # Forward pass to get predictions
        pred = model(
            pi_data.reshape(1, pi_data.shape[0], -1).to(device),
            aux.reshape(1, aux.shape[0], -1).to(device),
            imu.reshape(1, imu.shape[0], -1).to(device)
        ).cpu()

        # Calculate RMSE loss for each dimension (frontal, transversal, sagittal)
        loss = torch.nn.MSELoss()
        try:
            rmses = [torch.sqrt(loss(pred.squeeze()[dim], true[dim])).item() for dim in range(3)]
            # Calculate normalized RMSE
            nrmse = [rmses[dim] / (torch.max(true[dim]).item() - torch.min(true[dim]).item()) for dim in
                     range(3)]
            # Append RMSE and normalized RMSE values to the lists
            rmse_list.append(rmses)
            nrmse_list.append(nrmse)
        except ZeroDivisionError as e:
            # Print error message if there's a zero division error
            print('Error in calculation of values for subject %s', cfg.getint("experiment", "val_subj"))
            print("Error type: %s , Message: %s", type(e).__name__, str(e))

        # Convert lists to tensors
        rmse_tensor = torch.tensor(rmse_list)
        nrmse_tensor = torch.tensor(nrmse_list)

        # Compute mean RMSE and mean normalized RMSE over all samples and dimensions
        return torch.mean(rmse_tensor, 0), torch.mean(nrmse_tensor, 0)
