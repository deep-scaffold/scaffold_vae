"""Layer for probability flow"""
from abc import ABCMeta
from abc import abstractmethod
import math

import numpy as np
import torch
from torch import nn
# noinspection PyPep8Naming
from torch.nn import functional as F

import base


__all__ = ['MAF']


class Flow(nn.Module):
    """
    A abstract class for probability flow models
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def flow(self, *inputs):
        """Implements the forward flow, used during sampling"""
        raise NotImplementedError

    @abstractmethod
    def inverse(self, *inputs):
        """
        Implement the backward flow (or, the inverse flow), used for
        calculating likelihood
        """
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, *inputs):
        """
        Perform the inverse (backward) flow, and calculate the log-likelihood
        difference
        """
        raise NotImplementedError

    def forward(self, *inputs):
        pass


class MAFLayer(Flow):
    """
    Implements the basic building block of masked auto-regressive flow (MAF)
    """

    def __init__(self,
                 num_features,
                 num_replications,
                 activation='elu',
                 mean_only=False,
                 num_cond_features=None):
        """
        Initializer
        Args:
            num_features (int):
                The number of input features
            num_replications (list or tuple):
                How many times the features are being replicated at each layer
            activation (str): The activation function to use, default to 'elu'
            mean_only (bool):
                Whether to use mean-only affine transformation,
                default to `False`
            num_cond_features (int or None):
                The number of conditional features accepted,
                default to `None` (an unconditional model)
        """
        # Calling parent constructor
        super(MAFLayer, self).__init__()

        # Saving parameters
        self.num_features = num_features
        self.num_replications = list(num_replications)
        self.activation = activation
        self.mean_only = mean_only
        self.num_cond_features = num_cond_features

        if self.mean_only:
            self.num_replications = self.num_replications + [1, ]
        else:
            self.num_replications = self.num_replications + [2, ]

        # Building submodules
        self.masked_mlp = base.MaskedMLP(self.num_features,
                                         self.num_replications,
                                         self.activation,
                                         self.num_cond_features)

    def _eval(self,
              # pylint: disable=invalid-name
              x,
              # pylint: disable=invalid-name
              c=None):
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
            Tuple[torch.Tensor, torch.Tensor]:
                The result for mu and var
        """
        x_out = self.masked_mlp(x, c)

        if self.mean_only:
            mean = x_out.squeeze(-1)
            var = torch.ones_like(mean)
        else:
            mean, var = x_out[:, :, 0], x_out[:, :, 1]
            var = F.softplus(var) + 5e-5
        return mean, var

    def inverse(self, x, c=None):
        """
        Implement the backward flow (or, the inverse flow), used for
        calculating likelihood
        Args:
            x (torch.Tensor):
                The input tensor,
                dtype: `torch.float32`, shape: batch_size, num_features
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            torch.Tensor:
                The inverse projection of x
        """
        # The function mu and var
        mean, var = self._eval(x, c)
        std = torch.sqrt(var)
        x_out = (x - mean) / std
        return x_out

    def flow(self, x, c=None):
        """
        Implements the forward flow, used during sampling
        Args:
            x (torch.Tensor):
                The input tensor,
                dtype: `torch.float32`, shape: batch_size, num_features
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            torch.Tensor:
                The forward projection of x
        """
        x_out = torch.zeros_like(x)
        for i in range(self.num_features):
            mean, var = self._eval(x_out, c)
            std = torch.sqrt(var)
            x_out[:, i] = mean[:, i] + std[:, i] * x[:, i]
        return x_out

    def likelihood(self, x, c=None):
        """
        Perform the inverse (backward) flow, and calculate the log-likelihood
        difference
        Args:
            x (torch.Tensor):
                The input tensor,
                dtype: `torch.float32`, shape: batch_size, num_features
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The inverse projection of x and the ll difference
        """
        mean, var = self._eval(x, c)
        std = torch.sqrt(var)
        # pylint: disable=invalid-name
        ll = - 0.5 * torch.sum(torch.log(var), dim=-1)
        x_out = (x - mean) / std
        return x_out, ll


class Shuffle(Flow):
    """
    Shuffle the last dimension
    """
    def __init__(self, indices, dim=-1):
        """
        Initializer
        Args:
            indices (np.ndarray):
                The indices to shuffle the last dimension
            dim (int):
                The dimension to shuffle, default to the last dim
        """
        # Call parent constructor
        super(Shuffle, self).__init__()

        # Store parameter
        self.dim = dim

        # Register buffers
        num_features = indices.shape[0]
        self.register_buffer('indices',
                             torch.ones((num_features, ),
                                        dtype=torch.long))
        self.register_buffer('indices_inv',
                             torch.ones((num_features, ),
                                        dtype=torch.long))
        self._set_indices(indices)

    def _set_indices(self, indices):
        """
        Register indices
        Args:
            indices (np.ndarray): The indices to register
        """
        indices = indices.astype(np.int64)
        indices_inv = np.argsort(indices).astype(np.int64)

        self.indices.data.copy_(torch.from_numpy(indices))
        self.indices_inv.data.copy_(torch.from_numpy(indices_inv))

    def flow(self, x):
        """
        Implements the forward flow, used during sampling
        Args:
            x (torch.Tensor):
                The input tensor,
        Returns:
            torch.Tensor:
                The forward projection of x
        """
        return x.index_select(self.dim, self.indices)

    def inverse(self, x):
        """
        Implement the backward flow (or, the inverse flow), used for
        calculating likelihood
        Args:
            x (torch.Tensor):
                The input tensor,
        Returns:
            torch.Tensor:
                The inverse projection of x
        """
        return x.index_select(self.dim, self.indices_inv)

    def likelihood(self, x):
        """
        Perform the inverse (backward) flow, and calculate the log-likelihood
        difference
        Args:
            x (torch.Tensor):
                The input tensor
        Returns:
            Tuple[torch.Tensor, float]:
                The inverse projection of x and the ll difference
        """
        return self.inverse(x), 0.0


class Reverse(Shuffle):
    """
    Reverse the last dimension
    """

    def __init__(self, num_features, dim=-1):
        """
        Initializer
        Args:
            num_features (int): The number of features inside the input
            dim (int): The dimension to reverse, default to -1 (the last dim)
        """
        indices = np.arange(num_features, dtype=np.int64)[::-1]
        super(Reverse, self).__init__(indices, dim)


class BatchNormFlow(Flow):
    """Bidirectional batch normalization"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Initializer
        Args:
            num_features (int): The number of input features
            eps (float): Epsilon
            momentum (float): Momentum
        """
        super(BatchNormFlow, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # The weight before the softplus layer
        self.pre_weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        self.register_buffer('running_mean',
                             torch.zeros(num_features))
        self.register_buffer('running_var',
                             torch.ones(num_features))
        # pylint: disable=not-callable
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long))

        self.reset_parameters()

    # pylint: disable=missing-docstring
    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        # pylint: disable=no-member
        self.num_batches_tracked.zero_()

    # pylint: disable=missing-docstring
    def reset_parameters(self):
        self.reset_running_stats()
        self.pre_weight.data.fill_(math.log(math.e - 1.0))
        self.bias.data.zero_()

    @property
    def weight(self):
        """Return the weight after softplus"""
        return F.softplus(self.pre_weight) + self.eps

    def inverse(self, x):
        assert x.size(-1) == self.num_features, \
            'Input size should be {} ' \
            'instead of {}'.format(self.num_features,
                                   x.size(-1))

        exponential_average_factor = 0.0

        if self.training:
            # pylint: disable=no-member
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / \
                                             self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(x,
                            self.running_mean,
                            self.running_var,
                            self.weight,
                            self.bias,
                            self.training,
                            exponential_average_factor,
                            self.eps)

    def flow(self, x):
        assert x.size(-1) == self.num_features, \
            'Input size should be {} ' \
            'instead of {}'.format(self.num_features, x.size(-1))

        x_hat = (x - self.bias) / self.weight
        x = torch.sqrt(self.running_var + self.eps) * x_hat + self.running_mean
        return x

    def likelihood(self, x):
        assert x.size(-1) == self.num_features, \
            'Input size should be {} ' \
            'instead of {}'.format(self.num_features, x.size(-1))

        if self.training:
            mean = torch.mean(x, dim=0, keepdim=True)
            # pylint: disable=invalid-name
            v = torch.mean((x - mean) ** 2, dim=0) + self.eps
        else:
            # pylint: disable=invalid-name
            v = self.running_var + self.eps
        # pylint: disable=invalid-name
        ll = (- 0.5 * torch.sum(torch.log(v), dim=-1) +
              torch.sum(torch.log(self.weight), dim=-1))
        return self.inverse(x), ll

    def extra_repr(self):
        return ('{num_features}, '
                'eps={eps}, '
                'momentum={momentum}'.format(**self.__dict__))


class MAF(Flow):
    """Implements masked auto-regressive flow"""

    def __init__(self,
                 num_features,
                 num_layers,
                 num_replications,
                 num_cond_features=None,
                 activation='elu',
                 batch_norm=True,
                 mean_only=False):
        """
        Initializer
        Args:
            num_features (int):
                The number of input features
            num_layers (int):
                The number of `MAFLayers` to use
            num_replications (list or tuple):
                How many times the features are being replicated at each layer
            activation (str): The activation function to use, default to 'elu'
            mean_only (bool):
                Whether to use mean-only affine transformation,
                default to `False`
            batch_norm (bool):
                Whether to use `BatchNormFlow`, default to `True`
            num_cond_features (int or None):
                The number of conditional features accepted,
                default to `None` (an unconditional model)
        """
        # Calling parent constructor
        super(MAF, self).__init__()

        # Save parameters
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_replications = list(num_replications)
        self.num_cond_features = num_cond_features
        self.activation = activation
        self.mean_only = mean_only
        self.batch_norm = batch_norm

        # Building submodules
        layers = []
        for i in range(self.num_layers):
            if i != 0:
                layers.append(Reverse(self.num_features))
                if batch_norm:
                    layers.append(BatchNormFlow(self.num_features))
            layers.append(MAFLayer(self.num_features,
                                   self.num_replications,
                                   self.activation,
                                   self.mean_only,
                                   self.num_cond_features))
        self.layers = nn.ModuleList(layers)

    def inverse(self, x, c=None):
        """
        Implement the backward flow (or, the inverse flow), used for
        calculating likelihood
        Args:
            x (torch.Tensor):
                The input tensor,
                dtype: `torch.float32`, shape: batch_size, num_features
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            torch.Tensor:
                The inverse projection of x
        """
        x_out = x
        layers = [layer for layer in self.layers]
        for layer in reversed(layers):
            if isinstance(layer, MAFLayer):
                x_out = layer.inverse(x_out, c)
            else:
                x_out = layer.inverse(x_out)

        return x_out

    def flow(self, x, c=None):
        """
        Implements the forward flow, used during sampling
        Args:
            x (torch.Tensor):
                The input tensor,
                dtype: `torch.float32`, shape: batch_size, num_features
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            torch.Tensor:
                The forward projection of x
        """
        x_out = x
        for layer in self.layers:
            if isinstance(layer, MAFLayer):
                x_out = layer.flow(x_out, c)
            else:
                x_out = layer.flow(x_out)
        return x_out

    def likelihood(self, x, c=None):
        """
        Perform the inverse (backward) flow, and calculate the log-likelihood
        difference
        Args:
            x (torch.Tensor):
                The input tensor,
                dtype: `torch.float32`, shape: batch_size, num_features
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The inverse projection of x and the ll difference
        """
        x_out = x
        # pylint: disable=invalid-name
        ll = 0.0
        layers = [layer for layer in self.layers]
        for layer in reversed(layers):
            if isinstance(layer, MAFLayer):
                x_out, ll_i = layer.likelihood(x_out, c)
            else:
                x_out, ll_i = layer.likelihood(x_out)
            # pylint: disable=invalid-name
            ll = ll + ll_i

        return x_out, ll
