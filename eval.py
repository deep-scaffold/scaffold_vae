import sys
from os import path, makedirs
import json

import torch
from torch.utils.tensorboard import SummaryWriter
from ipypb import ipb
# import numpy
import multiprocess as mp
from joblib import Parallel, delayed

from data_utils import *
from mol_spec import MoleculeSpec
from scaffold_inference_network_architecture import *

# config_id = 'naive3'
# device_id = 2
# scaffolds_file = 'data-center/scaffolds_a.smi'

ms = MoleculeSpec.get_default()


def engine(
    config_id='naive3',
    device_id=3,
    model_idx=4,
    scaffolds_file='data-center/test.smi',
    batch_size=500,
    np=mp.cpu_count(),
):
    device = torch.device(f'cuda:{device_id}')
    model_ckpt = path.join(
        path.dirname(__file__),
        'ckpt',
        config_id,
        f'model_{model_idx}_ckpt.ckpt'
    )
    model_dic_loc = path.join(
        path.dirname(__file__),
        'ckpt',
        config_id,
        'modle_dic.json'
    )
    if path.exists(model_dic_loc):
        with open(model_dic_loc) as f:
            model_dic = json.load(f)
    else:
        model_dic = dict(
            num_in_feat=43,
            num_c_feat=8,
            num_embeddings=8,
            casual_hidden_sizes=[16, 32],
            num_botnec_feat=72,  # 16 x 4
            num_k_feat=24,  # 16
            num_dense_layers=20,
            num_out_feat=268,
            num_z_feat=10,
            activation='elu',
            use_cuda=True
        )
    # print(model_ckpt)
    model = GraphInf(**model_dic)
    model.load_state_dict(torch.load(model_ckpt))
    print(device_id)
    model.to(device)
    model.eval()

    dataloader = ComLoader(
        original_scaffolds_file=scaffolds_file,
        batch_size=batch_size,
        num_workers=1
    )

    all_num_valid = 0
    all_num_recon = 0
    events_loc = f'eval_configs/{config_id}/'
    if not path.exists(events_loc):
        makedirs(events_loc)

    with SummaryWriter(events_loc) as writer:
        step = 0
        with open(f'eval_configs/{config_id}_records.txt', 'w') as f:
            for batch in ipb(
                dataloader,
                desc="step",
                total=dataloader.num_id_block
            ):
                (
                    block,
                    nums_nodes,
                    nums_edges,
                    seg_ids,
                    bond_info_all,
                    nodes_o,
                    nodes_c
                ) = batch
                num_N = sum(nums_nodes)
                num_E = sum(nums_edges)

                values = torch.ones(num_E)

                s_adj = torch.sparse_coo_tensor(
                    bond_info_all.T,
                    values,
                    torch.Size([num_N, num_N])
                ).to(device)

                s_nfeat = torch.from_numpy(nodes_o).to(device)
                c_nfeat = torch.from_numpy(nodes_c).to(device)

                x_inf, mu2, var2 = model.inf(
                    c_nfeat,
                    s_adj
                )
                x_inf, mu2, var2 = (
                    x_inf.cpu().detach(),
                    mu2.cpu().detach(),
                    var2.cpu().detach()
                )
                x_recon, mu1, var1 = model.reconstrcut(
                    s_nfeat,
                    c_nfeat,
                    s_adj
                )
                x_recon, mu1, var1 = (
                    x_recon.cpu().detach(),
                    mu1.cpu().detach(),
                    var1.cpu().detach()
                )

                seg_ids = torch.from_numpy(seg_ids)

                MSE, KL = loss_func(
                    x_recon,
                    s_nfeat,
                    mu1,
                    var1,
                    mu2,
                    var2,
                    seg_ids
                )
                loss = MSE + KL
                writer.add_scalar(
                    f'loss',
                    loss.cpu().item(),
                    step
                )
                writer.add_scalar(
                    f'recon_loss',
                    MSE.cpu().item(),
                    step
                )
                writer.add_scalar(
                    f'KL',
                    KL.cpu().item(),
                    step
                )

                ls_x_inf = torch.split(x_inf, nums_nodes)
                ls_x_recon = torch.split(x_recon, nums_nodes)
                ls_mols_inf = Parallel(
                    n_jobs=np,
                    backend='multiprocessing'
                )(
                    delayed(get_mol_from_array)
                    (
                        ls_x_inf[i], block[i], True, False
                    )
                    for i in range(len(block))
                )
                ls_mols_recon = Parallel(
                    n_jobs=np,
                    backend='multiprocessing'
                )(
                    delayed(get_mol_from_array)
                    (
                        ls_x_recon[i], block[i], True, True
                    )
                    for i in range(len(block))
                )
                num_valid = sum(x is not None for x in ls_mols_inf)
                num_recon = sum(
                    ls_mols_recon[i] == block[i] for i in range(len(block))
                )
                all_num_valid += num_valid
                all_num_recon += num_recon
                f.write(
                    str(num_valid) + '\t' +
                    str(num_recon) + '\t' +
                    str(len(ls_mols_inf)) + '\n'
                )
                f.flush()
                step += 1

    with open(f'eval_configs/{config_id}.txt', 'w') as f:
        f.write(str(all_num_valid) + '\t')
        f.write(str(all_num_recon))


def main(config_id):
    "Program entrypoint"
    with open(
        path.join(
            path.dirname(__file__),
            'eval_configs',
            f'{config_id}.json',
        )
    ) as f:
        config = json.load(f)
        config['config_id'] = config_id
        engine(**config)


if __name__ == '__main__':
    main(sys.argv[1])
