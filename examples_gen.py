import sys
from os import path
import json

import torch
# from ipypb import ipb
# import numpy
# import multiprocess as mp
# from joblib import Parallel, delayed

from data_utils import *
from mol_spec import MoleculeSpec
from scaffold_inference_network_architecture import *

ms = MoleculeSpec.get_default()


def engine(
    config_id='naive3',
    device='cpu',
    c_file='data-center/scaffolds_a.smi',
    scaffold_idx=4,
    model_idx=3,
    num_gen=5,
):
    device = torch.device(device)
    model_ckpt = path.join(
        path.dirname(__file__),
        'ckpt',
        config_id,
        f'model_{model_idx}.ckpt'
    )
    model = torch.load(model_ckpt, map_location=device).eval()

    smiles = str_from_line(
        c_file,
        scaffold_idx
    )
    g, cg = smiles_to_dgl_graph(smiles)
    _, whole_cg = graph_to_whole_graph(g), graph_to_whole_graph(cg)

    num_N = whole_cg.number_of_nodes()
    num_E = whole_cg.number_of_edges()
    adj = whole_cg.adjacency_matrix().coalesce()
    indices = torch.cat(
        [adj.indices(), torch.arange(0, num_N).repeat(2, 1)],
        dim=-1
    )
    values = torch.ones(num_E + num_N)
    s_adj = torch.sparse_coo_tensor(
        indices,
        values,
        torch.Size([num_N, num_N])
    )
    c_nfeat = whole_cg.ndata['feat'].to(device)
    c_nfeat = onehot_to_label(c_nfeat)

    x_inf = model.inf(c_nfeat, s_adj).cpu().detach()

    mols = get_mols_from_array(
        x_inf,
        smiles,
        num_get=num_gen,
        to_smiles=True
    )

    with open(f'examples/{scaffold_idx}/mols.txt', 'w') as f:
        f.write(smiles + '\n')
        for mol_smiles in mols:
            f.write(mol_smiles + '\n')


def main(scaffold_idx):
    "Program entrypoint"
    with open(f'examples/{scaffold_idx}/config.json') as f:
        config = json.load(f)
        config['scaffold_idx'] = int(scaffold_idx)
        engine(**config)


if __name__ == '__main__':
    main(sys.argv[1])
