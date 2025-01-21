import torch
import torch.nn as nn
import torch.nn.functional as F
from CONT.utils.utils import get_sensors


class PiImutoMomentsNet(nn.Module):
    """
        A neural network model designed for processing PI and IMU data to predict 3d knee moments.
        Slope information is used as an auxiliary input

        Args:
            cfg (ConfigParser): Configuration object containing model parameters.

        Attributes:
            cfg (ConfigParser): Configuration object containing model parameters.

            pi_size_in (int): Input dimension for PI data (mean pressure of 7 segments + CoP Coordinates).
            pi_size_out (int): Output dimension for PI Layer
            pi_kernel (int): PI Layer Kernel Size
            pi_padding (int): PI Layer Padding

            imu_size_in (int): Input dimension for IMU data (3d accelerometer and gyroscope values of 4 IMUs)
                               depends on the number of sensors used (specified in config)
            imu_size_out (int): Output dimension for IMU Layer
            imu_kernel (int): IMU Layer Kernel Size
            imu_padding (int): IMU Layer Padding

            merge_size_in (int): Input dimension for merge layer
            merge_size_out (int): Output dimension for merge Layer
            merge_kernel (int): merge Layer Kernel Size
            merge_padding (int): merge Layer Padding

            aux_size_in (int): Input Dimension for Aux Layers
            aux_size_out (int): Output Dimension for Aux Layers
            hidden_dim (int): Number of Hidden Units in Aux Layers

            conv1_size_in (int): Input dimension for conv1 layer
            conv1_size_out (int): Output dimension for conv1 Layer
            conv1_kernel (int): conv1 Layer Kernel Size
            conv1_padding (int): conv1 Layer Padding

            x1_size_in (int): Input dimension for Extra layer 1
            x1_size_out (int): Output dimension for Extra layer 1
            x1_kernel (int): Extra layer 1 Kernel Size
            x1_padding (int): Extra layer 1 Layer Padding

            x2_size_in (int): Input dimension for Extra layer 2
            x2_size_out (int): Output dimension for Extra layer 2
            x2_kernel (int): Extra layer 2 Kernel Size
            x2_padding (int): Extra layer 2 Layer Padding

            out_size_in (int): Input dimension for Output layer
            out_size_out (int): Output dimension for Output layer (=3)
            out_kernel (int): Output Layer Kernel Size
            out_padding (int): Output Layer Padding

            conv_pi (nn.Conv1d): Convolutional layer for Pi data.
            conv_imu (nn.Conv1d): Convolutional layer for IMU data.
            merge (nn.Conv1d, optional): Convolutional layer for merged data - optional
            fc1 (nn.Linear): First fully connected layer for auxiliary data.
            fc2 (nn.Linear): Second fully connected layer for auxiliary data.
            conv1 (nn.Conv1d): Convolutional layer after concatenation and auxiliary processing.
            x1 (nn.Conv1d, optional): Convolutional Extra Layer 1 - optional
            x2 (nn.Conv1d, optional): Convolutional Extra Layer 2 - optional
            out (nn.Conv1d, optional): Final convolutional layer for generating output.
            dropout (nn.Dropout, optional): Dropout layer if dropout is enabled in the configuration.

        Methods:
            forward(pi, aux, imu): Forward pass through the network.
        """
    def __init__(self, cfg):
        super(PiImutoMomentsNet, self).__init__()

        self.cfg = cfg

        # PI layer
        self.pi_size_in = 9
        self.pi_size_out = self.cfg.getint('model_parameters', 'out_dim_pi_imu')
        self.pi_kernel = self.cfg.getint('model_parameters', 'kernel_in')
        self.pi_padding = (self.pi_kernel - 1) // 2

        # IMU layer
        self.imu_size_in = len(get_sensors(cfg)[0])
        self.imu_size_out = self.cfg.getint('model_parameters', 'out_dim_pi_imu')
        self.imu_kernel = self.cfg.getint('model_parameters', 'kernel_in') * 4  # multiplied by 4 - higher sampling rate
        self.imu_padding = ((self.cfg.getint('model_parameters', 'kernel_in') - 1) // 2) * 4

        # merge layer
        if self.cfg.getboolean('sensor_setup', 'pi') == False or self.imu_size_in == 0:
            self.merge_size_in = self.cfg.getint('model_parameters', 'out_dim_pi_imu')
        else:
            self.merge_size_in = self.cfg.getint('model_parameters', 'out_dim_pi_imu') * 2
        self.merge_size_out = self.merge_size_in
        self.merge_kernel = self.cfg.getint('model_parameters', 'kernel_merge')
        self.merge_padding = (self.merge_kernel - 1) // 2

        # Aux Layers
        self.aux_size_in = self.merge_size_out + 2
        self.aux_size_out = self.merge_size_out
        self.hidden_dim = self.cfg.getint('model_parameters', 'hidden_dim')

        # Conv Layer 1
        self.conv1_size_in = self.aux_size_out
        self.conv1_size_out = self.cfg.getint('model_parameters', 'conv1_dim')
        self.conv1_kernel = self.cfg.getint('model_parameters', 'kernel_conv1')
        self.conv1_padding = (self.conv1_kernel - 1) // 2

        # Conv Layer 2
        self.x1_size_in = self.conv1_size_out
        self.x1_size_out = self.cfg.getint('model_parameters', 'x1_dim')
        self.x1_kernel = self.cfg.getint('model_parameters', 'kernel_x1')
        self.x1_padding = (self.x1_kernel - 1) // 2

        # Conv Layer 3
        self.x2_size_in = self.x1_size_out
        self.x2_size_out = self.cfg.getint('model_parameters', 'x2_dim')
        self.x2_kernel = self.cfg.getint('model_parameters', 'kernel_x2')
        self.x2_padding = (self.x2_kernel - 1) // 2

        # output layer
        if self.cfg.getint('experiment', 'extra_layers') == 0:
            self.out_size_in = self.conv1_size_out
        elif self.cfg.getint('experiment', 'extra_layers') == 1:
            self.out_size_in = self.x1_size_out
        elif self.cfg.getint('experiment', 'extra_layers') == 2:
            self.out_size_in = self.x2_size_out
        else:
            raise ValueError("too many extra layer in configuration.. maximum = 2")
        self.out_size_out = self.cfg.getint('model_parameters', 'out_dim')
        self.out_kernel = self.cfg.getint('model_parameters', 'kernel_out')
        self.out_padding = (self.out_kernel - 1) // 2

        # PI Layer
        if self.cfg.getboolean("sensor_setup", "pi"):
            self.conv_pi = nn.Conv1d(in_channels=self.pi_size_in,
                                     out_channels=self.pi_size_out,
                                     groups=1,
                                     kernel_size=self.pi_kernel,
                                     padding=self.pi_padding)

        # IMU Layer
        if self.imu_size_in != 0:
            self.conv_imu = nn.Conv1d(in_channels=self.imu_size_in,
                                      out_channels=self.imu_size_out,
                                      groups=1,
                                      kernel_size=self.imu_kernel,
                                      padding=self.imu_padding,
                                      stride=4)  # stride = 4 to downsample

        # merge layer - specified in config
        if self.cfg.getboolean("experiment", "merge_layer"):
            self.merge = nn.Conv1d(in_channels=self.merge_size_in,
                                   out_channels=self.merge_size_out,
                                   groups=max(1, int(self.merge_size_in//16)),
                                   kernel_size=self.merge_kernel,
                                   padding=self.merge_padding)

        # Auxilliary layers
        self.fc1 = nn.Linear(in_features=self.aux_size_in,
                             out_features=self.hidden_dim)
        self.fc2 = nn.Linear(in_features=self.hidden_dim,
                             out_features=self.aux_size_out)

        # conv 1 layer
        self.conv1 = nn.Conv1d(in_channels=self.conv1_size_in,
                               out_channels=self.conv1_size_out,
                               groups=max(1, int(self.conv1_size_in//16)),
                               kernel_size=self.conv1_kernel,
                               padding=self.conv1_padding)

        # Extra Layer 1
        if self.cfg.getint('experiment', 'extra_layers') >= 1:
            self.x1 = nn.Conv1d(in_channels=self.x1_size_in,
                                out_channels=self.x1_size_out,
                                groups=max(1, int(self.x1_size_in/16)),
                                kernel_size=self.x1_kernel,
                                padding=self.x1_padding)

            # Extra Layer 1
            if self.cfg.getint('experiment', 'extra_layers') == 2:
                self.x2 = nn.Conv1d(in_channels=self.x2_size_in,
                                    out_channels=self.x2_size_out,
                                    groups=max(1, int(self.x2_size_in/16)),
                                    kernel_size=self.x2_kernel,
                                    padding=self.x2_padding)

        # output layer
        self.out = nn.Conv1d(in_channels=self.out_size_in,
                             out_channels=self.out_size_out,
                             groups=1,
                             kernel_size=self.out_kernel,
                             padding=self.out_padding)

        # dropout
        if self.cfg.getboolean("experiment", "dropout"):
            self.dropout = nn.Dropout(self.cfg.getfloat("model_parameters", "drop_ratio"))

    def forward(self, pi, aux, imu):
        """
                Forward pass through the network.

                Args:
                    pi (torch.Tensor): PI data tensor.
                    aux (torch.Tensor): Auxiliary data tensor (slope information).
                    imu (torch.Tensor): IMU data tensor.

                Returns:
                    torch.Tensor: Predicted output tensor.
                """
        # Feed in both data sources to separate conv1 layers - if respective sensors are set to True in config
        if self.cfg.getboolean('sensor_setup', 'pi'):
            # pi data
            x_pi = F.relu(self.conv_pi(pi))
            # dropout - if specified
            if self.cfg.getboolean("experiment", "dropout"):
                x_pi = self.dropout(x_pi)
            # imu data
            if self.imu_size_in != 0:
                x_imu = F.relu(self.conv_imu(imu))
                # dropout - if specified
                if self.cfg.getboolean("experiment", "dropout"):
                    x_imu = self.dropout(x_imu)

                # join & mix pi and imu so 1st dim looks like [x_pi[0], x_imu[0], x_pi[1], x_imu[1]...]
                x = torch.cat((x_pi.unsqueeze(2), x_imu.unsqueeze(2)), dim=2)
                x = x.view(x_pi.shape[0], self.merge_size_in, x_pi.shape[2])
            else:
                x = x_pi
        elif self.imu_size_in != 0:
            x = F.relu(self.conv_imu(imu))
            # dropout - if specified
            if self.cfg.getboolean("experiment", "dropout"):
                x = self.dropout(x)
        else:
            raise ValueError("All sensors set to False - please review sensor setup")

        # merge layer - if specified
        if self.cfg.getboolean("experiment", "merge_layer"):
            x = F.relu(self.merge(x))
        # dropout - if specified
        if self.cfg.getboolean("experiment", "dropout"):
            x = self.dropout(x)

        # concat and reshape to maintain time information while passing through auxiliary
        merged = torch.cat((x, aux), 1)
        merged = merged.transpose(1, 2)

        # Aux 1
        out = F.relu(self.fc1(merged))
        # dropout - if specified
        if self.cfg.getboolean("experiment", "dropout"):
            out = self.dropout(out)
        # Aux 2
        out = F.relu(self.fc2(out))
        # dropout - if specified
        if self.cfg.getboolean("experiment", "dropout"):
            out = self.dropout(out)
        out = torch.tanh(out)
        # reshape back
        x = out.transpose(1, 2)

        # Conv 1 Layer
        x = F.relu(self.conv1(x))
        # dropout - if specified
        if self.cfg.getboolean("experiment", "dropout"):
            x = self.dropout(x)

        # Extra Layer 1 - if specified
        if self.cfg.getint("experiment", "extra_layers") >= 1:
            x = F.relu(self.x1(x))
            # dropout - if specified
            if self.cfg.getboolean("experiment", "dropout"):
                x = self.dropout(x)

            # Extra Layer 2 - if specified
            if self.cfg.getint("experiment", "extra_layers") == 2:
                x = F.relu(self.x2(x))
                # dropout - if specified
                if self.cfg.getboolean("experiment", "dropout"):
                    x = self.dropout(x)

        # output layer
        x = self.out(x)

        return x
