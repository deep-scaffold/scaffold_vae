import typing as t

import torch
from torch import nn
from torch.nn import functional as F
# from torch import functional as F
import torch_scatter

from mol_spec import *

__all__ = [
    'loss_func',
    'get_activation'
]

ms = MoleculeSpec.get_default()


def loss_func(recon_x, x, mu1, var1, mu2, var2, seg_ids):
    num_atom_types = ms.num_atom_types
    num_bond_types = ms.num_bond_types

    is_atom = x.lt(num_atom_types)
    is_bond = (~ is_atom) & x.lt(num_atom_types + num_bond_types)
    is_virtual = x.ge(num_atom_types + num_bond_types)

    recon_x[is_atom, num_atom_types:] = - float('inf')
    recon_x[is_bond, :num_atom_types] = -float('inf')
    recon_x[is_bond, (num_atom_types + num_bond_types):] = -float('inf')
    recon_x = F.log_softmax(recon_x, dim=-1)

    seg_ids = seg_ids.to(mu1.device)
    # loss_recon = nn.CrossEntropyLoss().to(recon_x.device)
    # rec_loss = loss_recon(recon_x, x)
    num_total_nodes = recon_x.size(0)
    row_ids = torch.arange(num_total_nodes).long()
    rec_loss = - recon_x[row_ids, x.long()]
    rec_loss[is_virtual] = 0.0
    rec_loss = torch_scatter.scatter_add(
        rec_loss, seg_ids, dim=0
    ).mean()

    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = torch.sum(KLD_element).mul_(-0.5)

    # KLD_element = (
    #     (logvar2 - logvar1).mul(1 / 2).add(
    #         (
    #             logvar1.exp().add((mu1 - mu2).pow(2))
    #         ).div(2 * logvar2.exp()) -
    #         1 / 2
    #     )
    # )

    KLD_element = var2.div(var1).log().mul(0.5) + (
        var1 + (mu1 - mu2).pow(2)
    ).div(2 * var2) - 0.5
    KLD = torch_scatter.scatter_add(
        KLD_element, seg_ids, dim=0
    ).sum(dim=-1).mean()
    # KLD = KLD_element.mean()

    # KL divergence
    return rec_loss, KLD


def get_activation(
    name: str,
    *args,
    **kwargs
) -> t.Callable:
    """ Get activation module by name

    Args:
        name (str): The name of the activation function (relu, elu, selu)
        args, kwargs: Other parameters

    Returns:
        nn.Module: The activation module
    """
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(*args, **kwargs)
    elif name == 'elu':
        return nn.ELU(*args, **kwargs)
    elif name == 'selu':
        return nn.SELU(*args, **kwargs)
    else:
        raise ValueError('Activation not implemented')
