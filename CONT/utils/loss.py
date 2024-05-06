import torch
import torch.nn as nn


class AxisWiseLoss_RMSE(nn.Module):
    """ 
    Loss that weights each axis separately. Due to different ranges of moments in the different planes (sagittal, frontal, transversal).
    Calculate the axis-wise loss between ground truth and pred of shape (BATCHSIZE, n_channels, time).
    The loss is based on the torch.nn.MSELoss() and the squareroot is taken at the end to get the RMSE.

    Args:
        weights (tuple, optional): Different weight assigned to each plane. Default is (1, 1, 1).

    Returns:
        torch.Tensor: Mean RMSE over all planes

    """

    def __init__(self, weights=(1, 1, 1)):
        super(AxisWiseLoss_RMSE, self).__init__()
        self.weights = weights
        self.base_losses = [torch.nn.MSELoss() for _ in weights]

    def forward(self, moments, pred):
        """
        Compute the loss between ground truth and predicted moments.

        Args:
            moments (torch.Tensor): Ground truth moments of shape (BATCHSIZE, n_channels, time).
            pred (torch.Tensor): Predicted moments of shape (BATCHSIZE, n_channels, time).

        Returns:
            torch.Tensor: Mean RMSE over all planes
        """
        assert moments.size() == pred.size(), f"Shapes are: {moments.size()}, {pred.size()}"
        assert moments.size(-2) == len(self.weights)
        cum_sum = 0.0
        for idx, weight in enumerate(self.weights):
            # Calculate RMSE loss for each plane
            cum_sum += weight * torch.sqrt(self.base_losses[idx](
                moments.permute(1, 0, 2).reshape(len(self.weights), -1)[idx],
                pred.permute(1, 0, 2).reshape(len(self.weights), -1)[idx]
            ))
        # Return mean RMSE over all planes
        return cum_sum / len(self.weights)
