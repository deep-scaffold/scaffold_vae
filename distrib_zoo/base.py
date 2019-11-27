"""Several layers that are shared by all modules"""
import numpy as np
import torch
from torch import nn
# noinspection PyPep8Naming
from torch.nn import functional as F

import ops


__all__ = ['MaskedLinear',
           'SplitLinear',
           'BnActivationLinear',
           'MaskedMLP',
           'Lambda']


class MaskedLinear(nn.Linear):
    """`nn.Linear` with mask"""

    def __init__(self, in_features, out_features, mask, bias=True):
        """
        Initializer
        Args:
            in_features (int): The number of input features
            out_features (int): The number of output features
            mask (np.ndarray): The mask for the linear layer
            bias (bool): Whether to include bias, default to True
        """
        super(MaskedLinear, self).__init__(in_features, out_features, bias)

        mask = np.copy(mask)

        self.register_buffer('mask', torch.ones(out_features, in_features))
        self._set_mask(mask)

    def _set_mask(self, mask):
        """
        Register the input mask
        Args:
            mask (np.ndarray): The mask for the linear layer
        """
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, x):
        """
        The forward pass
        Args:
            x (torch.Tensor):
                The input tensor to be transformed,
                dtype: torch.float32, shape: [-1, in_features]
        Returns:
            torch.Tensor:
                The output tensor after the linear transformation
                dtype: torch.float32, shape: [-1, out_features]
        """
        return F.linear(x, self.mask * self.weight, self.bias)


class SplitLinear(MaskedLinear):
    """
    A special type of `MaskedLinear` layer.
    This layer split both the input and output dimension into half, denoted
    respectively as A, B and A', B'. The output A' only depends on the input A,
    but the output B' depends on both A and B.
    """

    def __init__(self, in_features, in_mask, out_features, out_mask, bias=True):
        """
        The initializer
        Args:
            in_features (int): The dimension of input
            in_mask (np.ndarray):
                The mask for the input
                dtype: np.bool, shape: [in_features]
                The A part of input is masked as `True`, others as `False`
            out_features (int): The dimension of output
            out_mask (np.ndarray):
                The mask for the output
                dtype: np.bool, shape: [out_features]
                The A' part of output is masked as `True`, others as `False`
            bias (bool): Whether to include bias, default to True
        """
        in_mask, out_mask = np.copy(in_mask), np.copy(out_mask)

        # Assert that `in_mask` and `out_mask` are both `np.ndarray` with
        # data type `np.bool`
        assert (isinstance(in_mask, np.ndarray)
                and in_mask.dtype == np.bool
                and isinstance(out_mask, np.ndarray)
                and out_mask.dtype == np.bool)

        # Build mask
        mask = np.zeros((in_features, out_features), dtype=np.float32)
        mask[in_mask, :] = 1.0
        mask[:, ~out_mask] = 1.0

        super(SplitLinear, self).__init__(in_features,
                                          out_features,
                                          mask, bias)


class BnActivationLinear(nn.Module):
    """Batch Normalization + Activation + Linear"""

    def __init__(self, in_features, out_features, activation='relu',
                 mask=None, in_mask=None, out_mask=None):
        """
        Initialization
        Args:
            in_features (int): The number of input features
            out_features (int): The number of output features
            mask (np.ndarray): The mask for the linear layer
            activation (str): The type of activation to use
        """
        super(BnActivationLinear, self).__init__()

        self.bn_relu_linear = nn.Sequential(nn.BatchNorm1d(in_features),
                                            ops.get_activation(activation,
                                                               inplace=True))
        if mask is not None:
            self.bn_relu_linear.add_module('linear',
                                           MaskedLinear(in_features,
                                                        out_features,
                                                        mask,
                                                        bias=False))
        elif in_mask is not None:
            assert out_mask is not None
            self.bn_relu_linear.add_module('linear',
                                           SplitLinear(in_features,
                                                       in_mask,
                                                       out_features,
                                                       out_mask,
                                                       bias=False))
        else:
            self.bn_relu_linear.add_module('linear',
                                           nn.Linear(in_features,
                                                     out_features,
                                                     bias=False))

    def forward(self, x):
        """
        The forward pass
        Args:
            x (torch.Tensor):
                The input tensor to be transformed,
                dtype: torch.float32, shape: [-1, in_features]
        Returns:
            torch.Tensor:
                The output tensor after the transformation
                dtype: torch.float32, shape: [-1, out_features]
        """
        return self.bn_relu_linear(x)


class MaskedMLP(nn.Module):
    """Masked multiple linear perceptron, used by `MADE` and `MAFLayer`"""

    def __init__(self,
                 num_features,
                 num_replications,
                 activation='elu',
                 num_cond_features=None):
        """
        Initializer
        Args:
            num_features (int):
                The number of input features
            num_replications (list or tuple):
                How many times the features are being replicated at each layer
            activation (str): The activation function to use, default to 'elu'
            num_cond_features (int or None):
                The number of conditional features accepted,
                default to `None` (an unconditional model)
        """
        # Calling parent constructor
        super(MaskedMLP, self).__init__()

        # Saving parameters
        self.num_features = num_features
        self.num_replications = list(num_replications)
        self.activation = activation
        self.num_cond_features = num_cond_features

        # Building submodules
        layers = []
        for i in range(len(self.num_replications)):
            # Get the size of output from the previous layer
            if i == 0:
                num_replication_prev = 1
            else:
                num_replication_prev = self.num_replications[i - 1]
            in_features = self.num_features * num_replication_prev

            # The the size of output for the current layer
            num_replication_next = self.num_replications[i]
            out_features = self.num_features * num_replication_next

            layer_i = []

            # Build mask
            # pylint: disable=invalid-name
            m = np.ones([self.num_features,
                         num_replication_prev,
                         self.num_features,
                         num_replication_next], dtype=np.float32)

            # Handle the differences between the last layer and other layers
            if i == len(self.num_replications) - 1:
                for j in range(self.num_features):
                    m[j:, :, j:(j + 1), :] = 0
            else:
                for j in range(self.num_features):
                    m[(j + 1):, :, j:(j + 1), :] = 0

            # Reshape mask
            # pylint: disable=invalid-name
            m = m.reshape([self.num_features * num_replication_prev,
                           self.num_features * num_replication_next])

            # Expand mask if conditional inputs are provided
            if self.num_cond_features is not None:
                m_c = np.ones([self.num_cond_features,
                               self.num_features * num_replication_next],
                              dtype=np.float32)
                # size: num_features * num_replication_prev + num_cond_features,
                #       num_features * num_replication_next
                # pylint: disable=invalid-name
                m = np.concatenate((m, m_c), axis=0)
                in_features += self.num_cond_features

            # batch_norm + activation
            layer_i.append(nn.BatchNorm1d(in_features))
            if i != 0:
                layer_i.append(ops.get_activation(self.activation,
                                                  inplace=True))

            # masked linear layer
            layer_i.append(MaskedLinear(in_features,
                                        out_features,
                                        m, bias=False))

            # merge
            layer_i = nn.Sequential(*layer_i)
            layers.append(layer_i)

        self.layers = nn.ModuleList(layers)

        # Add a batch normalization layer to the final output
        output_size = self.num_features * self.num_replications[-1]
        self.bn_final = nn.BatchNorm1d(output_size)

    def forward(self, x, c=None):
        """
        The feed forward network that parametrizes the function
        mu_i = mu(x[:i-1]) and var_i = var(x[:i-1])
        Args:
            x (torch.Tensor):
                The input tensor,
                dtype: `torch.float32`, shape: batch_size, num_features
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            torch.Tensor:
                dtype: `torch.float32`,
                shape: batch_size, num_features, num_replications[-1]
        """
        # Asserting that the input shapes are valid
        assert x.size(-1) == self.num_features, \
            'Input size should be {} instead of {}'.format(self.num_features,
                                                           x.size(-1))
        if self.num_cond_features is not None:
            assert c is not None, \
                'Conditional input should be provided'
            assert c.size(-1) == self.num_cond_features, \
                'The size of conditional input should be ' \
                '{} instead of {}'.format(self.num_cond_features,
                                          c.size(-1))

        # Rename input
        # size: batch_size, num_features
        x_out = x

        # Feed the input into each subsequent layers
        for layer in self.layers:
            # If the conditional code is specified, add to the input
            if c is not None:
                # size: batch_size, num_features * num_rep_prev + num_cond
                x_out = torch.cat([x_out, c], dim=-1)
            # size: batch_size, num_features * num_rep_next
            x_out = layer(x_out)

        # Perform the final step of batch_norm
        # size: batch_size, num_features (x num_rep[-1])
        x_out = self.bn_final(x_out)

        # Reshaping
        x_out = x_out.view(-1, self.num_features, self.num_replications[-1])

        return x_out


class Lambda(nn.Module):
    """
    Wrap function `fn` into a pytorch module
    """
    def __init__(self,
                 # pylint: disable=invalid-name
                 fn):
        """
        Args:
            fn: A callable to be wrapped
        """
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, *inputs):
        """
        The forward pass
        """
        return self.fn(*inputs)
