"""
SAPE_EEGNet by Nadine El-Mufit, based on the original EEGNet from https://doi.org/10.1088/1741-2552/aace8c.

The original EEGNet is a shallow and lightweight convolutional neural network proposed for a general decoding of single-trial EEG signals,
suitable for applications such as P300, error-related negativity, motor execution, and motor imagery decoding.

This modified version employs sequence-adaptive sinusoidal positional embeddings to enhance temporal accuracy in EEG signal processing,
ideal for tasks requiring precise time-series understanding.

Original Author:
 * Davide Borra, 2021

Modifications by:
 * Nadine El-Mufit, 2024
"""

import torch
import speechbrain as sb
import numpy as np


class SAPE_EEGNet(torch.nn.Module):
    """
        SAPE_EEGNet.
        
        Description
        ---------
        (Sequence-Adaptive Positional Encoding EEGNet) is an enhanced version of the traditional EEGNet, 
        specifically tailored to improve the processing of EEG signals through advanced positional encoding techniques. 
        This model utilizes sinusoidal positional embeddings, which are crucial for capturing the temporal dynamics 
        inherent in EEG data. The embeddings differentiate positions within the sequence by applying sine functions 
        to even indices and cosine functions to odd indices, allowing the model to maintain an awareness of the 
        temporal order of inputs.

        This architecture is designed to better handle tasks that require understanding of time-series data, such as 
        decoding P300, error-related negativity, motor execution, and motor imagery from single-trial EEG signals.

        Arguments
        ---------
        input_shape : tuple
            Specifies the shape of the input tensor expected by the model, formatted as 
            (Batch size, Time steps, Channels, 1).
        cnn_temporal_kernels : int
            The number of convolutional kernels used in the temporal convolution layer.
        cnn_temporal_kernelsize : tuple
            The size of the kernels for the temporal convolution.
        cnn_spatial_depth_multiplier : int
            The multiplier for the number of output channels from the spatial depthwise convolution layer.
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
        >>> model = SAPE_EEGNet(input_shape=inp_tensor.shape)
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
        elif activation_type == "selu": # New Activation Function
            activation = torch.nn.SELU()
        elif activation_type == "mish": # New Activation Function
            activation = torch.nn.Mish()
        elif activation_type == "swish": # New Activation Function
            activation = torch.nn.Hardswish()        
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
        Generate sinusoidal positional embeddings with a (unique) approach: using sine for even positions
        and cosine for odd positions. This method applies the trigonometric functions across all dimensions 
        for a given position based on its order in the sequence, enhancing the model's temporal resolution.

        Parameters:
        length (int): The temporal length of the sequence.
        d_model (int): The dimensionality of the embeddings.
        device (torch.device): The device to generate the embeddings on.

        Returns:
        torch.Tensor: The positional embeddings with shape (length, d_model).
        """
        positional_embedding = torch.zeros((length, d_model), device=device)
        # Omega: scaling factor for adjusting frequencies of the sine and cosine functions.
        omega = 10000 ** (torch.arange(0, d_model, 2, device=device).float() / d_model)
        
        pos_range = torch.arange(length, device=device).unsqueeze(1)  
        even_indices = torch.arange(0, d_model, 2, device=device)
        odd_indices = torch.arange(1, d_model, 2, device=device)

        # Apply sine to even positions and cosine to odd positions across all positions
        positional_embedding[:, 0::2] = torch.sin(pos_range * omega / d_model)
        positional_embedding[:, 1::2] = torch.cos(pos_range * omega / d_model)

        return positional_embedding

    def forward(self, x):
        # Step 1: Apply the first convolutional layer to perform temporal convolution
        x = self.conv_module[0](x)

        # Step 2: Calculate and apply positional embeddings to the output of the first convolution
        temporal_length = x.shape[2]  # Extract the number of temporal steps from the feature map
        d_model = x.shape[1]  # Extract the number of channels as the dimensionality of the embeddings
        pos_embeddings = self.generate_positional_embeddings(temporal_length, d_model, x.device)
        pos_embeddings = pos_embeddings.unsqueeze(0).unsqueeze(-1)  # Adjust the shape for broadcasting
        
        # Reshape positional embeddings to match the shape of the convolution output
        pos_embeddings = pos_embeddings.reshape(1, 500, 17, 1)  # Example reshape to match 'x' dimensions

        # Add positional embeddings to enhance the feature map with temporal information
        x += pos_embeddings

        # Step 3: Apply batch normalization to stabilize and normalize the outputs
        x = self.conv_module[1](x)

        # Step 4: Continue processing with the subsequent layers in the convolutional module
        for layer in self.conv_module[2:]:
            x = layer(x)

        # Step 5: Final processing through the dense module to prepare the final output
        x = self.dense_module(x)
        
        return x
