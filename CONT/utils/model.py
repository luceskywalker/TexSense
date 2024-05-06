import torch
import torch.nn as nn
import torch.nn.functional as F

class PiImutoMomentsNet(nn.Module):
    """
        A neural network model designed for processing PI and IMU data to predict 3d knee moments.
        Slope information is used as an auxiliary input

        Args:
            cfg (ConfigParser): Configuration object containing model parameters.

        Attributes:
            cfg (ConfigParser): Configuration object containing model parameters.
            in_dim_pi (int): Input dimension for PI data (mean pressure of 7 segments + CoP Coordinates).
            in_dim_imu (int): Input dimension for IMU data (3d accelerometer and gyroscope values of 4 IMUs)
            h1_dim (int): Dimension of the first layer.
            join_dim (int): Dimension after concatenation of Pi and IMU data.
            h2_dim (int): Dimension of the second layer.
            out_dim (int): Output dimension.
            kernel_dim1 (int): Dimension of the first convolutional kernel.
            kernel_dim2 (int): Dimension of the second convolutional kernel.
            kernel_dim3 (int): Dimension of the third convolutional kernel.
            padding_dim1 (int): Padding dimension for the first convolutional layer.
            padding_dim2 (int): Padding dimension for the second convolutional layer.
            padding_dim3 (int): Padding dimension for the third convolutional layer.
            conv1_pi (nn.Conv1d): Convolutional layer for Pi data.
            conv1_imu (nn.Conv1d): Convolutional layer for IMU data.
            conv1_join (nn.Conv1d): Convolutional layer for concatenated data.
            hidden_dim (int): Dimension of the hidden layer in auxiliary fully connected layers.
            fc1 (nn.Linear): First fully connected layer for auxiliary data.
            fc2 (nn.Linear): Second fully connected layer for auxiliary data.
            conv2 (nn.Conv1d): Convolutional layer after concatenation and auxiliary processing.
            conv3 (nn.Conv1d): Final convolutional layer for generating output.
            dropout (nn.Dropout, optional): Dropout layer if dropout is enabled in the configuration.

        Methods:
            forward(pi, aux, imu): Forward pass through the network.
        """
    def __init__(self, cfg):
        super(PiImutoMomentsNet, self).__init__()

        # Initialize model parameters from configuration
        self.cfg = cfg

        self.in_dim_pi = self.cfg.getint('model_parameters', 'in_dim_pi')
        self.in_dim_imu = self.cfg.getint('model_parameters', 'in_dim_imu')
        self.h1_dim = self.cfg.getint('model_parameters', 'h1_dim')
        self.join_dim = self.h1_dim * 2
        self.h2_dim = self.cfg.getint('model_parameters', 'h2_dim')
        self.out_dim = self.cfg.getint('model_parameters', 'out_dim')

        self.kernel_dim1 = self.cfg.getint('model_parameters', 'kernel_dim1')
        self.kernel_dim2 = self.cfg.getint('model_parameters', 'kernel_dim2')
        self.kernel_dim3 = self.cfg.getint('model_parameters', 'kernel_dim3')


        self.padding_dim1 = self.cfg.getint('model_parameters', 'padding_dim1')
        self.padding_dim2 = self.cfg.getint('model_parameters', 'padding_dim2')
        self.padding_dim3 = self.cfg.getint('model_parameters', 'padding_dim3')


        # PI layer 1
        self.conv1_pi = nn.Conv1d(in_channels=self.in_dim_pi,       
                                  out_channels=self.h1_dim,         
                                  groups=1,
                                  kernel_size=self.kernel_dim1,     
                                  padding=self.padding_dim1)        

        # IMU layer 1
        self.conv1_imu = nn.Conv1d(in_channels=self.in_dim_imu,     
                                   out_channels = self.h1_dim,      
                                   groups=1,
                                   kernel_size=self.kernel_dim1 * 4,
                                   padding=self.padding_dim1 * 4,   
                                   stride=4)                        
        
        # Join Inputs layer
        self.conv1_join = nn.Conv1d(in_channels = self.join_dim,    
                                    out_channels = self.join_dim,   
                                    groups=16,
                                    kernel_size=self.kernel_dim1,   
                                    padding=self.padding_dim1)      

        # AUX Layer
        self.hidden_dim = self.cfg.getint('model_parameters', 'hidden_dim')
        self.fc1 = nn.Linear(in_features=self.join_dim + 2,   	
                                out_features=self.hidden_dim)      
        self.fc2 = nn.Linear(in_features=self.hidden_dim,       
                                out_features=self.join_dim)     

        # conv layer 2
        self.conv2 = nn.Conv1d(in_channels=self.join_dim,         	
                               out_channels=self.h2_dim,            
                               groups=16,
                               kernel_size=self.kernel_dim2,        
                               padding=self.padding_dim2)           


        # output layer
        self.conv3 = nn.Conv1d(in_channels=self.h2_dim,             
                               out_channels=self.out_dim,           
                               groups=1,
                               kernel_size=self.kernel_dim3,        
                               padding=self.padding_dim3)           

        # dropout
        if self.cfg.getboolean("experiment", "dropout"):
            self.dropout = nn.Dropout(self.cfg.getfloat("model_parameters", "drop_ratio"))

    def forward(self, pi, aux, imu):
        """
                Forward pass through the network.

                Args:
                    pi (torch.Tensor): Pi data tensor.
                    aux (torch.Tensor): Auxiliary data tensor (slope information).
                    imu (torch.Tensor): IMU data tensor.

                Returns:
                    torch.Tensor: Predicted output tensor.
                """
        # Feed in both data sources to separate conv1 layers
        x_pi = F.relu(self.conv1_pi(pi))     
        x_imu = F.relu(self.conv1_imu(imu))    

        # join & mix pi and imu so 1st dim looks like [x_pi[0], x_imu[0], x_pi[1], x_imu[1]...]
        x = torch.cat((x_pi.unsqueeze(2), x_imu.unsqueeze(2)), dim=2)     
        x = x.view(x_pi.shape[0], self.join_dim, x_pi.shape[2])
        x = F.relu(self.conv1_join(x))         
        
        # dropout layer
        if self.cfg.getboolean("experiment", "dropout"):
            x = self.dropout(x)

        # auxillary fc layers
        merged = torch.cat((x, aux), 1)     
        merged = merged.transpose(1, 2)   

        out = F.relu(self.fc1(merged))
        out = F.relu(self.fc2(out))
        out = torch.tanh(out)

        x = out.transpose(1, 2)

        # second conv layer
        x = F.relu(self.conv2(x))   

        # final conv layer
        x = self.conv3(x)                

        return x
