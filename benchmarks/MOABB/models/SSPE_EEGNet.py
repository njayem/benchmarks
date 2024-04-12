"""
SSPE_EEGNet by Nadine El-Mufit, based on the original EEGNet from https://doi.org/10.1088/1741-2552/aace8c.
The original EEGNet is a shallow and lightweight convolutional neural network proposed for a general decoding of single-trial EEG signals,
suitable for applications such as P300, error-related negativity, motor execution, and motor imagery decoding.
This version, named EEGNetMSNoTDrop, introduces modifications incorporating Mish and Swish activation functions and removing temporal dropout.
Original Author:
 * Davide Borra, 2021

Modifications by:
 * Nadine El-Mufit, 2024
"""

import torch
import speechbrain as sb
import numpy as np


class SSPE_EEGNet(torch.nn.Module):
    """
    SSPE_EEGNet.
    
    Description
    ---------
    (Standard Sinusoidal Positional Encoding EEGNet) is an enhancement of the original EEGNet 
    designed specifically for improved handling of EEG data by employing consistent sinusoidal positional embeddings. 
    This version applies a fixed pattern of sine for even and cosine for odd indices across all positions in the sequence, 
    which helps the network maintain an accurate perception of the temporal order throughout the EEG signal processing.

    This model is particularly suited for EEG applications such as P300, error-related negativity, motor execution, and 
    motor imagery decoding, where understanding the exact sequence of EEG data points is crucial.

    Arguments
    ---------
    input_shape : tuple
        Specifies the shape of the input tensor (Batch size, Time steps, Channels, 1).
    cnn_temporal_kernels : int
        The number of convolutional kernels used in the temporal convolution layer.
    cnn_temporal_kernelsize : tuple
        The size of the kernels for the temporal convolution.
    cnn_spatial_depth_multiplier : int
        Multiplier for the output channels from the spatial depthwise convolution.
    cnn_spatial_max_norm : float
        The maximum norm for the kernels in the spatial depthwise convolution, used for regularization.
    cnn_spatial_pool : tuple
        The dimensions for pooling (pool size and stride) following the spatial convolution.
    cnn_septemporal_depth_multiplier : int
        Multiplier for the output channels in the separable temporal convolution layers.
    cnn_septemporal_kernelsize : tuple
        The kernel size for the separable temporal convolution.
    cnn_septemporal_pool : tuple
        Pooling dimensions (pool size and stride) following the separable temporal convolution.
    cnn_pool_type : str
        Type of pooling layer ('avg' or 'max') used in the convolutional modules.
    dropout : float
        Dropout rate for regularization to prevent overfitting.
    dense_max_norm : float
        Maximum norm for the fully-connected layer weights.
    dense_n_neurons : int
        Number of neurons in the fully-connected output layer.
    activation_type : str
        Type of activation function used in hidden layers ('relu', 'elu', etc.).

    Example:
    -------
    >>> inp_tensor = torch.rand([1, 200, 32, 1])
    >>> model = SSPE_EEGNet(input_shape=inp_tensor.shape)
    >>> output = model(inp_tensor)
    >>> output.shape
    torch.Size([1, 4])
    """

    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        cnn_temporal_kernels=8,
        cnn_temporal_kernelsize=(33, 1),
        cnn_spatial_depth_multiplier=2,
        cnn_spatial_max_norm=1.0,
        cnn_spatial_pool=(4, 1),
        cnn_septemporal_depth_multiplier=1,
        cnn_septemporal_point_kernels=None,
        cnn_septemporal_kernelsize=(17, 1),
        cnn_septemporal_pool=(8, 1),
        cnn_pool_type="avg",
        dropout=0.5,
        dense_max_norm=0.25,
        dense_n_neurons=4,
        activation_type="elu",
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        if activation_type == "gelu":
            activation = torch.nn.GELU()
        elif activation_type == "elu":
            activation = torch.nn.ELU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        elif activation_type == "prelu":
            activation = torch.nn.PReLU()
        elif activation_type == "selu":
            activation = torch.nn.SELU()
        else:
            raise ValueError("Wrong hidden activation function")
       
        self.default_sf = 128  # sampling rate of the original publication (Hz)
        
        # T = input_shape[1]
        C = input_shape[2]

        # CONVOLUTIONAL MODULE
        self.conv_module = torch.nn.Sequential()
        
        # Temporal convolution
        self.conv_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_temporal_kernels, momentum=0.01, affine=True,
            ),
        )
        
        # Spatial depthwise convolution
        cnn_spatial_kernels = (
            cnn_spatial_depth_multiplier * cnn_temporal_kernels
        )
        self.conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_spatial_kernels,
                kernel_size=(1, C),
                groups=cnn_temporal_kernels,
                padding="valid",
                bias=False,
                max_norm=cnn_spatial_max_norm,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels, momentum=0.01, affine=True,
            ),
        )
        self.conv_module.add_module("act_1", activation)
        self.conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_spatial_pool,
                stride=cnn_spatial_pool,
                pool_axis=[1, 2],
            ),
        )
        self.conv_module.add_module("dropout_1", torch.nn.Dropout(p=dropout))

        # Temporal separable convolution
        cnn_septemporal_kernels = (
            cnn_spatial_kernels * cnn_septemporal_depth_multiplier
        )
        self.conv_module.add_module(
            "conv_2",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_spatial_kernels,
                out_channels=cnn_septemporal_kernels,
                kernel_size=cnn_septemporal_kernelsize,
                groups=cnn_spatial_kernels,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )

        if cnn_septemporal_point_kernels is None:
            cnn_septemporal_point_kernels = cnn_septemporal_kernels

        self.conv_module.add_module(
            "conv_3",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_septemporal_kernels,
                out_channels=cnn_septemporal_point_kernels,
                kernel_size=(1, 1),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_septemporal_point_kernels,
                momentum=0.01,
                affine=True,
            ),
        )
        self.conv_module.add_module("act_3", activation)
        self.conv_module.add_module(
            "pool_3",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_septemporal_pool,
                stride=cnn_septemporal_pool,
                pool_axis=[1, 2],
            ),
        )
        self.conv_module.add_module("dropout_3", torch.nn.Dropout(p=dropout))

        # Shape of intermediate feature maps
        out = self.conv_module(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        dense_input_size = self._num_flat_features(out)
        # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=dense_n_neurons,
                max_norm=dense_max_norm,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def _num_flat_features(self, x):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def generate_positional_embeddings(self, length, d_model, device):
    """
    Generate sinusoidal positional embeddings for the EEGNet model using a (standard) sinusoidal encoding approach.
    This method consistently applies sine functions to even indices and cosine functions to odd indices of the embedding 
    vector for each position. This consistent pattern across all positions helps the network to process temporal sequence 
    data effectively, crucial for accurate EEG signal analysis.

    Parameters:
    ----------
    length : int
        The number of time steps in the sequence for which embeddings are to be generated.
    d_model : int
        The number of dimensions of each embedding, representing the complexity of the encoding.
    device : torch.device
        The computing device (CPU, GPU) where the embeddings will be generated, affecting performance.

    Returns:
    -------
    torch.Tensor
        A tensor of shape (length, d_model), with each row representing the sinusoidal positional embeddings for a 
        specific time step in the sequence.

    Example:
    -------
    >>> embeddings = self.generate_positional_embeddings(200, 64, torch.device('cuda'))
    >>> embeddings.shape
    (200, 64)
    """
    position = torch.arange(length, dtype=torch.float).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).to(device)

    positional_embedding = torch.zeros((length, d_model), device=device)
    positional_embedding[:, 0::2] = torch.sin(position * div_term)

    if d_model % 2 == 1:  
        div_term_cos = torch.exp(torch.arange(0, d_model - 1, 2).float() * -(np.log(10000.0) / d_model)).to(device)
        positional_embedding[:, 1::2] = torch.cos(position * div_term_cos)
    else:
        positional_embedding[:, 1::2] = torch.cos(position * div_term)

        return positional_embedding

    def forward(self, x):
        """Returns the output of the model with positional embeddings added after the first temporal convolution."""

        # Step 1: Apply the first convolution layer
        x = self.conv_module[0](x)
        
        # Step 2: Apply batch normalization
        x = self.conv_module[1](x)

        # Step 3: Calculate the number of time steps and the model dimensions for positional embeddings
        temporal_length = x.shape[2]
        d_model = x.shape[3]

        # Step 4: Generate positional embeddings based on the current model configuration
        pos_embeddings = self.generate_positional_embeddings(temporal_length, d_model, x.device)

        # Step 5: Adjust the shape of positional embeddings for broadcasting compatibility
        pos_embeddings = pos_embeddings.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, Temporal, Features]
        
        # Step 6: Add positional embeddings to the convolution output to enrich features with positional information
        x += pos_embeddings
        
        # Step 7: Continue processing through the subsequent layers of the convolutional module
        for layer in self.conv_module[2:]:
            x = layer(x)

        # Step 8: Pass the output through the dense module to get the final model output
        x = self.dense_module(x)
        
        return x
