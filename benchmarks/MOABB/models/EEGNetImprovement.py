"""EEGNet from https://doi.org/10.1088/1741-2552/aace8c.
Shallow and lightweight convolutional neural network proposed for a general decoding of single-trial EEG signals.
It was proposed for P300, error-related negativity, motor execution, motor imagery decoding.

Authors
 * Davide Borra, 2021
"""
import torch
import speechbrain as sb


class EEGNetImprovement(torch.nn.Module):
    """EEGNet.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    cnn_temporal_kernels: int
        Number of kernels in the 2d temporal convolution.
    cnn_temporal_kernelsize: tuple
        Kernel size of the 2d temporal convolution.
    cnn_spatial_depth_multiplier: int
        Depth multiplier of the 2d spatial depthwise convolution.
    cnn_spatial_max_norm: float
        Kernel max norm of the 2d spatial depthwise convolution.
    cnn_spatial_pool: tuple
        Pool size and stride after the 2d spatial depthwise convolution.
    cnn_septemporal_depth_multiplier: int
        Depth multiplier of the 2d temporal separable convolution.
    cnn_septemporal_kernelsize: tuple
        Kernel size of the 2d temporal separable convolution.
    cnn_septemporal_pool: tuple
        Pool size and stride after the 2d temporal separable convolution.
    cnn_pool_type: string
        Pooling type.
    dropout: float
        Dropout probability.
    dense_max_norm: float
        Weight max norm of the fully-connected layer.
    dense_n_neurons: int
        Number of output neurons.
    activation_type: str
        Activation function of the hidden layers.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = EEGNet(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
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

        # Activation function selection
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
            raise ValueError("Unsupported activation function type.")

        # Temporal convolution x2 (back to back)
        # Spatial depthwise convolution x2 (back to back)
        # Followed by Temporal separable convolution (as in original)
        self.conv_module = torch.nn.Sequential(
            # First Temporal Convolution
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,
                padding="same",
                bias=False,
            ),
            sb.nnet.normalization.BatchNorm2d(cnn_temporal_kernels),
            activation,
            
            # Second Temporal Convolution
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,
                padding="same",
                bias=False,
            ),
            sb.nnet.normalization.BatchNorm2d(cnn_temporal_kernels),
            activation,

            # First Spatial Depthwise Convolution
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_temporal_kernels * cnn_spatial_depth_multiplier,
                kernel_size=(1, input_shape[2]),
                groups=cnn_temporal_kernels,  # Ensuring depthwise operation
                padding="valid",
                bias=False,
                max_norm=cnn_spatial_max_norm,
            ),
            sb.nnet.normalization.BatchNorm2d(cnn_temporal_kernels * cnn_spatial_depth_multiplier),
            activation,

            # Second Spatial Depthwise Convolution
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels * cnn_spatial_depth_multiplier,
                out_channels=cnn_temporal_kernels * cnn_spatial_depth_multiplier,
                kernel_size=(1, 1),
                groups=cnn_temporal_kernels * cnn_spatial_depth_multiplier,  # Ensuring depthwise operation
                padding="valid",
                bias=False,
                max_norm=cnn_spatial_max_norm,
            ),
            sb.nnet.normalization.BatchNorm2d(cnn_temporal_kernels * cnn_spatial_depth_multiplier),
            activation,

            # Pooling and Dropout as in the original configuration
            sb.nnet.pooling.Pooling2d(pool_type=cnn_pool_type, kernel_size=cnn_spatial_pool, stride=cnn_spatial_pool),
            torch.nn.Dropout(p=dropout),

            # Temporal Separable Convolution (Unchanged)
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels * cnn_spatial_depth_multiplier,
                out_channels=cnn_septemporal_kernels,
                kernel_size=cnn_septemporal_kernelsize,
                groups=cnn_temporal_kernels * cnn_spatial_depth_multiplier,  # Ensuring depthwise operation
                padding="same",
                bias=False,
            ),
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_septemporal_kernels,
                out_channels=cnn_septemporal_point_kernels or cnn_septemporal_kernels,
                kernel_size=(1, 1),
                padding="valid",
                bias=False,
            ),
            sb.nnet.normalization.BatchNorm2d(cnn_septemporal_point_kernels or cnn_septemporal_kernels),
            activation,
            sb.nnet.pooling.Pooling2d(pool_type=cnn_pool_type, kernel_size=cnn_septemporal_pool, stride=cnn_septemporal_pool),
            torch.nn.Dropout(p=dropout),
        )

        # Shape of intermediate feature maps to calculate dense input size dynamically
        example_input = torch.zeros((1,) + input_shape[1:])
        dense_input_size = self._num_flat_features(self.conv_module(example_input))

        # DENSE MODULE
        self.dense_module = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(dense_input_size, dense_n_neurons),
            torch.nn.LogSoftmax(dim=1)
        )

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.conv_module(x)
        x = self.dense_module(x)
        return x