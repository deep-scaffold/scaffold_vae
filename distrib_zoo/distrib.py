"""Probability distributions"""
from abc import ABCMeta, abstractmethod
import math

import torch
from torch import nn
# noinspection PyPep8Naming
from torch.nn import functional as F

import base
import flow


__all__ = ['Distribution', 'Gaussian', 'MADE2MAF']


class Distribution(nn.Module):
    """The base class for distributions"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def sample(self, *inputs):
        """Sample from the probability distribution"""
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, *inputs):
        """Calculate the likelihood"""
        raise NotImplementedError

    def forward(self, *inputs):
        pass


class Gaussian(Distribution):
    """The gaussian distribution"""

    def __init__(self, num_features, loc=0.0, scale=1.0):
        super(Gaussian, self).__init__()

        self.loc = loc
        self.scale = scale
        self.num_features = num_features

    def sample(self, num_samples, device=torch.device('cpu')):
        """
        Sampling from gaussian distribution
        Args:
            num_samples (int):
                The number of samples to sample
            device (torch.device):
                The device where the sample locates, default to cpu
        Returns:
            torch.Tensor:
                dtype: `torch.float32`, shape: [batch_size, num_features]
        """
        mean = self.loc * torch.ones([num_samples,
                                      self.num_features],
                                     dtype=torch.float32,
                                     device=device)
        std = self.scale * torch.ones([num_samples,
                                       self.num_features],
                                      dtype=torch.float32,
                                      device=device)
        return torch.normal(mean, std)

    def likelihood(self, x):
        """
        Get the likelihood of x
        Args:
            x (torch.Tensor):
                The sample to get the likelihood
                dtype: `torch.float32`, shape: [batch_size, num_features]
        Returns:
            torch.Tensor:
                The likelihood value calculated
                dtype: `torch.float32`, shape: [batch_size, ]
        """
        var = self.scale ** 2
        # pylint: disable=invalid-name
        mu = self.loc
        log_var = (torch.log(var)
                   if isinstance(var, torch.Tensor)
                   else math.log(var))
        loss = (- 0.5 * math.log(2 * math.pi)
                - 0.5 * log_var
                - (x - mu) ** 2 / (2.0 * var))
        return torch.sum(loss, axis=-1)


class MADE(Distribution):
    """Masked auto-encoder density estimation"""

    def __init__(self,
                 num_features,
                 num_gaussian,
                 num_replications,
                 activation='elu',
                 num_cond_features=None):
        """
        Initializer
        Args:
            num_features (int):
                The number of input features
            num_gaussian (int):
                The number of gaussian to use in the MoG output
            num_replications (list or tuple):
                How many times the features are being replicated at each layer
            activation (str): The activation function to use, default to 'elu'
            num_cond_features (int or None):
                The number of conditional features accepted,
                default to `None` (an unconditional model)
        """
        # Calling parent constructor
        super(MADE, self).__init__()

        # Saving parameters
        self.num_features = num_features
        self.num_gaussian = num_gaussian
        self.num_replications = list(num_replications)
        self.activation = activation
        self.num_cond_features = num_cond_features

        self.num_replications += [self.num_gaussian * 3, ]

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
        The feed forward network
        Args:
            x (torch.Tensor):
                The input tensor,
                dtype: `torch.float32`, shape: batch_size, num_features
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The result for mu, var and alpha
        """
        x_out = self.masked_mlp(x, c)

        # size: batch_size, num_features, num_gaussian, 3
        x_out = x_out.view(-1, self.num_features, self.num_gaussian, 3)
        # size: batch_size, num_features, num_gaussian
        # pylint: disable=invalid-name
        alpha, mu, var = (x_out[:, :, :, 0],
                          x_out[:, :, :, 1],
                          x_out[:, :, :, 2])

        var = F.softplus(var) + 5e-5

        # Normalize
        # size: batch_size, num_features, num_gaussian
        log_alpha = F.log_softmax(alpha, dim=-1)

        return log_alpha, mu, var

    def likelihood(self, x, c=None):
        """
        Get the likelihood of x
        Args:
            x (torch.Tensor):
                The sample to get the likelihood
                dtype: `torch.float32`, shape: [batch_size, num_features]
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            torch.Tensor:
                The likelihood value calculated
                dtype: `torch.float32`, shape: [batch_size, ]
        """
        # shape: batch_size, num_features, num_gaussian
        # pylint: disable=invalid-name
        log_alpha, mu, var = self._eval(x, c)
        # batch_size x num_features x 1
        x = x.unsqueeze(dim=-1)
        # batch_size x num_features x num_gaussian
        x = x.expand(-1, -1, self.num_gaussian)

        loss = (- 0.5 * math.log(2 * math.pi)
                - 0.5 * torch.log(var)
                - (x - mu) ** 2 / (2.0 * var))
        loss = loss + log_alpha

        # batch_size x num_features
        loss = torch.logsumexp(loss, dim=-1)

        loss = torch.sum(loss, dim=-1)
        return loss

    def sample(self, num_samples, c=None):
        """
        Sampling from gaussian distribution
        Args:
            num_samples (int):
                The number of samples to sample
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            torch.Tensor:
                dtype: `torch.float32`, shape: [batch_size, num_features]
        """
        # Get device information
        device = next(self.parameters()).device

        # Initialize output
        x_out = torch.zeros([num_samples,
                             self.num_features],
                            device=device,
                            dtype=torch.float32)

        if c is not None:
            if not (len(c.shape) in [1, 2]):
                raise ValueError('dim of c should be 1 or 2')
            if len(c.shape) == 1:
                # If c is a one dimensional tensor, then we replicate c to each
                # sample to be generated
                c = c.unsqueeze(0).expand(num_samples, -1)

        for i in range(self.num_features):
            # shape: batch_size, num_features, num_gaussian
            # pylint: disable=invalid-name
            log_alpha, mu, var = self._eval(x_out, c)
            # shape: batch_size, num_gaussian
            log_alpha_i, mu_i, var_i = (log_alpha[:, i, :],
                                        mu[:, i, :],
                                        var[:, i, :])

            # sample category
            # shape: batch_size,
            category = torch.multinomial(torch.exp(log_alpha_i), 1)[:, 0]
            index = torch.arange(category.size(0),
                                 dtype=torch.long,
                                 device=device)
            mu_i, var_i = mu_i[index, category], var_i[index, category]

            std_i = torch.sqrt(var_i)
            x_i = torch.normal(mu_i, std_i)
            x_out[:, i] = x_i
        return x_out


class MADE2MAF(Distribution):
    """The masked auto-regressive flow (MAF) with MADE prior"""

    def __init__(self,
                 num_features,
                 num_gaussian,
                 num_layers,
                 num_replications,
                 num_cond_features=None,
                 activation='elu'):
        """
        Initializer
        Args:
            num_features (int):
                The number of input features
            num_gaussian (int):
                The number of gaussian to use in the MoG output
            num_layers (int):
                The number of `MAFLayers` to use
            num_replications (list or tuple):
                How many times the features are being replicated at each layer
            activation (str): The activation function to use, default to 'elu'
            num_cond_features (int or None):
                The number of conditional features accepted,
                default to `None` (an unconditional model)
        """
        # Calling parent constructor
        super(MADE2MAF, self).__init__()

        # Building submodules
        self.made = MADE(num_features,
                         num_gaussian,
                         num_replications,
                         activation,
                         num_cond_features)
        self.maf = flow.MAF(num_features,
                            num_layers,
                            num_replications,
                            num_cond_features,
                            activation,
                            mean_only=False)

    def likelihood(self, x, c=None):
        """
        Get the likelihood of x
        Args:
            x (torch.Tensor):
                The sample to get the likelihood
                dtype: `torch.float32`, shape: [batch_size, num_features]
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            torch.Tensor:
                The likelihood value calculated
                dtype: `torch.float32`, shape: [batch_size, ]
        """
        # Inverse masked auto-regressive flow
        # shape: batch_size,
        x, ll_maf = self.maf.likelihood(x, c)
        # Likelihood from MADE
        # shape: batch_size,
        ll_made = self.made.likelihood(x, c)

        # pylint: disable=invalid-name
        ll = ll_maf + ll_made

        return ll

    def sample(self,
               num_samples,
               c=None):
        """
        Sampling from gaussian distribution
        Args:
            num_samples (int):
                The number of samples to sample
            c (torch.Tensor or None):
                The conditional inputs
                dtype: `torch.float32`, shape: batch_size, num_cond_features
        Returns:
            torch.Tensor:
                dtype: `torch.float32`, shape: [batch_size, num_features]
        """
        # Generate sample from MADE
        # pylint: disable=invalid-name
        x = self.made.sample(num_samples, c)
        # Map sample with MAF
        # pylint: disable=invalid-name
        x = self.maf.flow(x, c)
        return x
