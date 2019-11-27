import sys
import json
from os import path, makedirs
import typing as t

from ipypb import ipb
import torch
# from torch import nn
from torch.utils.tensorboard import SummaryWriter
import adabound

from scaffold_inference_network_architecture import *
from data_utils import *


def engine(
    ckpt_loc: str='ckpt-default',
    device_id: int=1,
    batch_size: int=128,
    use_cuda: bool=True,
    num_embeddings: int=8,
    casual_hidden_sizes: t.Iterable=[16, 32],
    num_botnec_feat: int=72,
    num_k_feat: int=24,
    num_dense_layers: int=20,
    num_out_feat: int=268,
    num_z_feat: int=10,
    activation: str='elu',
    LR: float=1e-3,
    final_lr: float=0.1,
    init_beta: float=0.,
    final_beta: float=1.,
    num_annealing_steps: int=2000,
    # beta: float=0.25,
    grad_clip=3.0,
    num_epochs: int=5,
    num_p=1
):
    beta_step_len = (final_beta - init_beta) / num_annealing_steps
    model = GraphInf(
        num_in_feat=43,
        num_c_feat=8,
        num_embeddings=num_embeddings,
        casual_hidden_sizes=casual_hidden_sizes,
        num_botnec_feat=num_botnec_feat,  # 16 x 4
        num_k_feat=num_k_feat,  # 16
        num_dense_layers=num_dense_layers,
        num_out_feat=num_out_feat,
        num_z_feat=num_z_feat,
        activation=activation,
        use_cuda=use_cuda
    )
    model_dic = dict(
        num_in_feat=43,
        num_c_feat=8,
        num_embeddings=num_embeddings,
        casual_hidden_sizes=casual_hidden_sizes,
        num_botnec_feat=num_botnec_feat,  # 16 x 4
        num_k_feat=num_k_feat,  # 16
        num_dense_layers=num_dense_layers,
        num_out_feat=num_out_feat,
        num_z_feat=num_z_feat,
        activation=activation,
        use_cuda=use_cuda
    )

    optim = adabound.AdaBound(
        model.parameters(),
        lr=LR,
        final_lr=final_lr
    )
    device = torch.device(f'cuda:{device_id}')
    model = model.to(device)
    model.train()
    save_loc = path.join(
        path.dirname(__file__),
        'ckpt',
        ckpt_loc
    )
    with open(path.join(save_loc, 'model_dic.json'), 'w') as f:
        f.write(json.dumps(model_dic))

    events_loc = path.join(save_loc, 'events')
    if not path.exists(events_loc):
        makedirs(events_loc)
    try:
        with SummaryWriter(events_loc) as writer:
            step = 0
            has_nan_or_inf = False
            train_loader = ComLoader(
                original_scaffolds_file='data-center/train.smi',
                num_workers=num_p,
                batch_size=batch_size
            )
            test_loader = ComLoader(
                original_scaffolds_file='data-center/test.smi',
                num_workers=num_p,
                batch_size=batch_size
            )

            for epoch in ipb(range(num_epochs), desc='epochs'):
                iter_train = iter(train_loader)
                iter_test = iter(test_loader)
                try:
                    if has_nan_or_inf:
                        break

                    for i in ipb(range(
                        train_loader.num_id_block +
                        train_loader.num_id_block // 200
                    ), desc='iteration'):
                        if step > 0 and step % 200 == 0:
                            batch = next(iter_test)
                        else:
                            batch = next(iter_train)

                        (
                            block,
                            nums_nodes,
                            nums_edges,
                            seg_ids,
                            bond_info_all,
                            nodes_o,
                            nodes_c
                        ) = batch

                        beta = min(init_beta + beta_step_len * step, 1)
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

                        x_recon, mu1, logvar1, mu2, logvar2 = (
                            model(s_nfeat, c_nfeat, s_adj)
                        )
                        seg_ids = torch.from_numpy(seg_ids)
                        optim.zero_grad()
                        MSE, KL = loss_func(
                            x_recon,
                            s_nfeat,
                            mu1,
                            logvar1,
                            mu2,
                            logvar2,
                            seg_ids
                        )

                        loss = MSE + beta * KL
                        if not (step > 0 and step % 200 == 0):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                grad_clip
                            )
                            optim.step()

                        # debug for Nan in recon loss
                        has_nan_or_inf = torch.cat(
                            [
                                torch.stack(
                                    (
                                        torch.isnan(params.grad).any(),
                                        torch.isinf(params.grad).any()
                                    ),
                                    dim=-1
                                ) for params in model.parameters()
                            ],
                            dim=-1
                        ).any()

                        if has_nan_or_inf:
                            torch.save(
                                model,
                                path.join(save_loc, f'broken_{epoch}.ckpt')
                            )
                            torch.save(
                                s_nfeat,
                                path.join(save_loc, f's_nfeat_{epoch}.pt')
                            )
                            torch.save(
                                c_nfeat,
                                path.join(save_loc, f'c_nfeat_{epoch}.pt')
                            )
                            torch.save(
                                s_adj.to_dense(),
                                path.join(save_loc, f's_adj_{epoch}.pt')
                            )
                            torch.save(
                                seg_ids,
                                path.join(save_loc, f'seg_ids_{epoch}.pt')
                            )
                            with open(
                                path.join(save_loc, 'batch.smi'), 'w'
                            ) as f:
                                for smiles in block:
                                    f.write(smiles + '\n')

                            break

                        if not (step > 0 and step % 200 == 0):
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
                        else:
                            writer.add_scalar(
                                f'test_loss',
                                loss.cpu().item(),
                                step
                            )
                            writer.add_scalar(
                                f'test_recon_loss',
                                MSE.cpu().item(),
                                step
                            )
                            writer.add_scalar(
                                f'test_KL',
                                KL.cpu().item(),
                                step
                            )

                        step += 1

                    torch.save(
                        model.state_dict(),
                        path.join(save_loc, f'model_{epoch}_ckpt.ckpt')
                    )
                except StopIteration:
                    continue

    except KeyboardInterrupt:
        torch.save(
            model,
            path.join(save_loc, f'model_{epoch}.ckpt')
        )


def main(ckpt_loc):
    "Program entrypoint"
    with open(
        path.join(
            path.dirname(__file__),
            'ckpt',
            ckpt_loc,
            'config.json'
        )
    ) as f:
        config = json.load(f)
        config['ckpt_loc'] = ckpt_loc
        engine(**config)


if __name__ == '__main__':
    main(sys.argv[1])
