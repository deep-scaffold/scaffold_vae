import linecache
import typing as t
from random import shuffle

import dgl
# import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy
from torch.nn import functional as F
from torch_sparse import spspmm

from mol_spec import *

__all__ = [
    'onehot_to_label',
    'label_to_onehot',
    'smiles_to_dgl_graph',
    'ms',
    'graph_from_line',
    'get_num_lines',
    'str_from_line',
    'graph_to_whole_graph',
    'get_remote_connection',
    'whole_graph_from_smiles',
    'str_block_gen',
    'get_mol_from_array',
    'get_mols_from_array',
    'spmmsp',

    # 'np_onehot_to_label'
]

ms = MoleculeSpec.get_default()

_file_name_to_content = {}


def _get_file_by_line(file_name: str, line_id: int):
    if file_name not in _file_name_to_content:
        with open(file_name) as f:
            file_content = tuple([line.rstrip() for line in f])
        _file_name_to_content[file_name] = file_content
    return _file_name_to_content[file_name][line_id]


# def block_id_gen(
#     file: str,
#     batch_size: int
# ):
#     num_lines = get_num_lines(file)
#     ids = list(range(num_lines))
#     shuffle(ids)
#     num_id_block = (
#         num_lines // batch_size if
#         num_lines % batch_size == 0 else
#         num_lines // batch_size + 1
#     )
#     id_block = [
#         ids[
#             i * batch_size:min(
#                 (i + 1) * batch_size, num_lines
#             )
#         ]
#         for i in range(num_id_block)
#     ]
#     return id_block


def str_block_gen(
    file: str,
    batch_size: int
):
    num_lines = get_num_lines(file)
    ids = list(range(num_lines))
    shuffle(ids)
    num_id_block = (
        num_lines // batch_size if
        num_lines % batch_size == 0 else
        num_lines // batch_size + 1
    )
    ids_block = [
        ids[
            i * batch_size:min(
                (i + 1) * batch_size, num_lines
            )
        ]
        for i in range(num_id_block)
    ]

    smiles_block = [
        [_get_file_by_line(file, i) for i in block] for block in ids_block
    ]

    return smiles_block

    # for block in ids_block:
    #     str_block = [_get_file_by_line(file, i) for i in block]
    #     yield str_block


def smiles_to_dgl_graph(
    smiles: str,
    ms: MoleculeSpec = ms,
    # ranked: bool=False  # wrong parameter name, whether it's a C scaffold
) -> t.Tuple:
    """Convert smiles to dgl graph
    Args:
        smiles (str): a molecule smiles
        ms (MoleculeSpec, optional): MoleculeSpec
    """
    # smiles = standardize_smiles(smiles)
    try:
        m = AllChem.MolFromSmiles(smiles)

        m = Chem.RemoveHs(m)
        g = dgl.DGLGraph()
        cg = dgl.DGLGraph()

        g.add_nodes(m.GetNumAtoms())
        cg.add_nodes(m.GetNumAtoms())

        ls_edge = [
            (e.GetBeginAtomIdx(), e.GetEndAtomIdx()) for e in m.GetBonds()
        ]

        ls_atom_type = [
            ms.get_atom_type(atom) for atom in m.GetAtoms()
        ]
        ls_edge_type = [
            ms.get_bond_type(bond) for bond in m.GetBonds()
        ]

        src, dst = tuple(zip(*ls_edge))

        g.add_edges(src, dst)
        g.add_edges(dst, src)

        cg.add_edges(src, dst)
        cg.add_edges(dst, src)

        # if ranked:
        cg.ndata['feat'] = label_to_onehot(
            torch.LongTensor([0 for _ in ls_atom_type]),
            1
        )
        cg.edata['feat'] = label_to_onehot(
            torch.LongTensor([0 for _ in ls_edge_type]),
            1
        ).repeat(2, 1)
        # else:
        g.ndata['feat'] = label_to_onehot(
            torch.LongTensor(ls_atom_type),
            len(ms.atom_types)
        )
        g.edata['feat'] = label_to_onehot(
            torch.LongTensor(ls_edge_type),
            len(ms.bond_orders)
        ).repeat(2, 1)
        return (g, cg)
    except:
        return (None, None)


def label_to_onehot(ls, class_num):
    ls = ls.reshape(-1, 1)
    return torch.zeros(
        (len(ls), class_num), device=ls.device
    ).scatter_(1, ls, 1)


def onehot_to_label(tensor):
    if isinstance(tensor, torch.Tensor):
        return torch.argmax(tensor, dim=-1)
    elif isinstance(tensor, numpy.ndarray):
        return numpy.argmax(tensor, axis=-1)


def str_from_line(
    file: str,
    idx: int
) -> str:
    """
    Get string from a specific line
    Args:
        idx (int): index of line
        file (string): location of a file

    Returns:

    """
    return linecache.getline(file, idx + 1).strip()


def get_num_lines(
    input_file: str
) -> int:
    """Get num_of_lines of a text file
    Args:
        input_file (str): location of the file

    Returns:
        int: num_lines of the file

    Examples:
        >>> get_num_lines("./dataset.txt")
    """
    for num_lines, line in enumerate(open(input_file, 'r')):
        pass
    return num_lines + 1


def graph_from_line(
    file: str,
    idx: int,
    ranked: bool=False
) -> dgl.DGLGraph:
    smiles = str_from_line(
        file,
        idx
    )
    g = smiles_to_dgl_graph(
        smiles,
        ms=ms,
        ranked=ranked
    )
    return g


def get_nonzero_idx(sp: torch.sparse.FloatTensor) -> torch.LongTensor:
    """Get indices of nonzero elements of a sparse tensor
    
    Args:
        sp (torch.sparse.FloatTensor): a sparse tensor
    
    Returns:
        torch.LongTensor: indices of nonzero elements
    """
    sp = sp.coalesce()
    return sp.indices()[:, sp.values() > 0]


def get_remote_connection(
    adj: torch.Tensor,
) -> t.Tuple[torch.Tensor, ...]:
    d = spmmsp(adj.coalesce(), adj.coalesce())
    # d_indices_2 = d.to_dense().nonzero().t()
    d_indices_2 = get_nonzero_idx(d)
    d_indices_2 = d_indices_2[:, d_indices_2[0, :] != d_indices_2[1, :]]
    d = spmmsp(d.coalesce(), adj.coalesce())
    d = d - d.mul(adj)
    # d_indices_3 = d.to_dense().nonzero().t()
    d_indices_3 = get_nonzero_idx(d)
    d_indices_3 = d_indices_3[:, d_indices_3[0, :] != d_indices_3[1, :]]
    return d_indices_2, d_indices_3


def graph_to_whole_graph2(
    adj: torch.Tensor,
    bond_info: torch.Tensor,
    n_feat: torch.Tensor,
    e_feat: torch.Tensor
) -> t.Tuple[torch.Tensor, ...]:
    """Involving remote connections and consider edges as nodes
   
    Args:
        adj (torch.Tensor):
            adj with out self connections N x N
        bond_info (torch.Tensor):
            original bond info 2 x N_e
        n_feat (torch.Tensor):
            original node feat N x F
        e_feat (torch.Tensor):
            original edge feat N_e x F_e
    """
    # adj = g.adjacency_matrix()
    # bond_info = torch.stack(g.edges(), dim=0)
    # n_feat = g.ndata['feat']
    # e_feat = g.edata['feat']

    num_n_feat = n_feat.size(-1)
    d_indices_2, d_indices_3 = get_remote_connection(adj)
    all_bond_info = torch.cat([bond_info, d_indices_2, d_indices_3], dim=-1)
    all_e_feat = torch.cat(
        [
            torch.cat(
                [
                    e_feat,
                    torch.zeros([e_feat.size(0), 2])
                ],
                dim=-1
            ),
            torch.cat(
                [
                    torch.zeros([d_indices_2.size(-1), e_feat.size(-1)]),
                    torch.ones([d_indices_2.size(-1), 1]),
                    torch.zeros([d_indices_2.size(-1), 1])
                ],
                dim=-1
            ),
            torch.cat(
                [
                    torch.zeros([d_indices_3.size(-1), e_feat.size(-1)]),
                    torch.zeros([d_indices_3.size(-1), 1]),
                    torch.ones([d_indices_3.size(-1), 1])
                ],
                dim=-1
            )
        ],
        dim=0
    )
    num_n = n_feat.size(0)
    # num_e = all_e_feat.size(0)
    ndata_new = torch.cat(
        (n_feat, torch.zeros(num_n, all_e_feat.size(1))),
        dim=1
    )
    edata_new = torch.cat(
        (torch.zeros([all_e_feat.size(0), num_n_feat]), all_e_feat),
        dim=1
    )
    all_node_data = torch.cat(
        (ndata_new, edata_new),
        dim=0
    )
    n_new = torch.arange(
        num_n,
        all_node_data.size(0)
    )
    all_new_bond_info = torch.cat(
        [
            torch.stack(
                [all_bond_info[0], n_new],
                dim=0
            ),
            torch.stack(
                [n_new, all_bond_info[0]],
                dim=0
            ),
            torch.stack(
                [all_bond_info[1], n_new],
                dim=0
            ),
            torch.stack(
                [n_new, all_bond_info[1]],
                dim=0
            )
        ],
        dim=-1
    )
    adj = (
        torch.eye(all_node_data.size(0)).to_sparse() +
        torch.sparse_coo_tensor(
            all_new_bond_info,
            [1. for _ in range(all_new_bond_info.size(-1))],
            torch.Size(
                [all_node_data.size(0), all_node_data.size(0)]
            )
        )
    )
    return onehot_to_label(all_node_data), all_new_bond_info, adj


def graph_to_whole_graph(
    g: dgl.DGLGraph
) -> dgl.DGLGraph:
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    adj = g.adjacency_matrix()
    d_indices_2, d_indices_3 = get_remote_connection(adj)
    e_data = g.edata['feat']
    g.add_edges(d_indices_2[0], d_indices_2[1])
    g.add_edges(d_indices_3[0], d_indices_3[1])
    g.edata['feat'] = torch.cat(
        [
            torch.cat([e_data, torch.zeros([e_data.size(0), 2])], dim=-1),
            torch.cat(
                [
                    torch.zeros([d_indices_2.size(-1), e_data.size(-1)]),
                    torch.ones([d_indices_2.size(-1), 1]),
                    torch.zeros([d_indices_2.size(-1), 1])
                ], dim=-1),
            torch.cat(
                [
                    torch.zeros([d_indices_3.size(-1), e_data.size(-1)]),
                    torch.zeros([d_indices_3.size(-1), 1]),
                    torch.ones([d_indices_3.size(-1), 1])
                ], dim=-1)
        ],
        dim=0
    )
    g_new = dgl.DGLGraph()
    g_new.add_nodes(g.number_of_nodes() + g.number_of_edges())
    n_add = torch.arange(g.number_of_nodes(), g_new.number_of_nodes())
    ndata_new = torch.cat(
        (
            g.ndata['feat'],
            torch.zeros((g.number_of_nodes(), g.edata['feat'].size(-1)))
        ),
        dim=-1
    )
    edata_new = torch.cat(
        (
            torch.zeros((g.edata['feat'].size(0), g.ndata['feat'].size(-1))),
            g.edata['feat']
        ),
        dim=-1
    )
    all_node_data = torch.cat(
        (ndata_new, edata_new),
        dim=0
    )
    g_new.ndata['feat'] = all_node_data
    all_new_bond_info = torch.cat(
        [
            torch.stack(
                [g.edges()[0], n_add],
                dim=0
            ),
            torch.stack(
                [n_add, g.edges()[0]],
                dim=0
            ),
            torch.stack(
                [g.edges()[1], n_add],
                dim=0
            ),
            torch.stack(
                [n_add, g.edges()[1]],
                dim=0
            ),

        ],
        dim=-1
    )
    g_new.add_edges(all_new_bond_info[0], all_new_bond_info[1])
    return g_new


def whole_graph_from_smiles(
    smiles: str,
) -> t.Tuple:
    g, cg = smiles_to_dgl_graph(smiles)
    return (graph_to_whole_graph(g), graph_to_whole_graph(cg))

# def get_whole_data(
#     g: dgl.graph.DGLGraph,
#     key_n: str='feat',
#     key_e: str='feat'
# ):
#     num_n_feat, num_e_feat = g.ndata[key_n].size(-1), g.edata[key_e].size(-1)
#     # num_feat = num_n_feat + num_e_feat
#     num_n, num_e = g.ndata[key_n].size(0), int(g.edata[key_e].size(0) / 2)
#     batch_size = num_n + num_e
#     ndata_new = torch.cat(
#         (g.ndata[key_n], torch.zeros(num_n, num_e_feat)),
#         dim=1
#     )
#     edata_new = torch.cat(
#         (torch.zeros([num_e, num_n_feat]), g.edata[key_e][: num_e]),
#         dim=1
#     )
#     all_node_data = torch.cat(
#         (ndata_new, edata_new),
#         dim=0
#     )

#     n_new = torch.arange(num_n, batch_size)
#     indices = torch.cat(
#         [n_new for _ in range(2)],
#         dim=-1
#     )
#     indices1 = torch.stack(
#         [indices, g.edges()[0]],
#         dim=0
#     )
#     indices2 = torch.stack(
#         [g.edges()[0], indices],
#         dim=0
#     )
#     indices = torch.cat([indices1, indices2], dim=-1)
#     n_e_adj = torch.sparse_coo_tensor(
#         indices,
#         [1. for _ in range(indices.size(-1))],
#         torch.Size([batch_size, batch_size])
#     )
#     adj = (
#         torch.eye(batch_size).to_sparse() +
#         n_e_adj
#     )

#     return all_node_data, adj


def get_mol_from_array(
    array,
    smiles,
    sanitize=True,
    to_smiles=False
):
    m = Chem.MolFromSmiles(smiles)
    num_atoms = m.GetNumAtoms()
    num_bonds = m.GetNumBonds()
    atom_array = array[: num_atoms, :33]
    bond_array = array[num_atoms: num_atoms + num_bonds, 33: 37]

    atom_array = atom_array - atom_array.max(dim=-1, keepdim=True).values
    bond_array = bond_array - bond_array.max(dim=-1, keepdim=True).values

    atom_array_e = F.softmax(atom_array, dim=-1)
    bond_array_e = F.softmax(bond_array, dim=-1)

    atom_array_e[atom_array_e < 0] = 0
    atom_array_e[atom_array_e != atom_array_e] = 1
    bond_array_e[bond_array_e < 0] = 0
    bond_array_e[bond_array_e != bond_array_e] = 1

    mol = get_mol_from_clean_array(
        atom_array_e,
        bond_array_e,
        smiles=smiles,
        sanitize=sanitize,
        to_smiles=to_smiles
    )

    return mol


def get_mol_from_clean_array(
    atom_array_e,
    bond_array_e,
    smiles,
    sanitize=True,
    to_smiles=True
):
    m = Chem.MolFromSmiles(smiles)

    atom_indices = torch.multinomial(atom_array_e, 1).flatten().tolist()
    bond_indices = torch.multinomial(bond_array_e, 1).flatten().tolist()

    mol = Chem.RWMol(Chem.Mol())
    for atom_idx in atom_indices:
        mol.AddAtom(ms.index_to_atom(atom_idx))
    for bond_idx, bond in zip(bond_indices, m.GetBonds()):
        ms.index_to_bond(
            mol,
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_idx
        )
    if sanitize:
        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            if to_smiles:
                return Chem.MolToSmiles(mol)
            else:
                return mol
        except:
            return None
    else:
        return mol


def get_mols_from_array(
    array: torch.Tensor,
    smiles: str,
    num_get: int,
    sanitize: bool=True,
    to_smiles: bool=False,
):
    m = Chem.MolFromSmiles(smiles)
    num_atoms = m.GetNumAtoms()
    num_bonds = m.GetNumBonds()
    atom_array = array[: num_atoms, :33]
    bond_array = array[num_atoms: num_atoms + num_bonds, 33: 37]

    atom_array = atom_array - atom_array.max(dim=-1, keepdim=True).values
    bond_array = bond_array - bond_array.max(dim=-1, keepdim=True).values

    atom_array_e = F.softmax(atom_array, dim=-1)
    bond_array_e = F.softmax(bond_array, dim=-1)

    atom_array_e[atom_array_e < 0] = 0
    atom_array_e[atom_array_e != atom_array_e] = 1
    bond_array_e[bond_array_e < 0] = 0
    bond_array_e[bond_array_e != bond_array_e] = 1

    mols = []
    while True:
        mol = get_mol_from_clean_array(
            atom_array_e,
            bond_array_e,
            smiles=smiles,
            sanitize=sanitize,
            to_smiles=to_smiles
        )
        if mol is not None:
            mols.append(mol)
        if len(mols) >= num_get:
            break
    return mols


def spmmsp(
    sp1: torch.Tensor,
    sp2: torch.Tensor
) -> torch.Tensor:
    assert sp1.size(-1) == sp2.size(0) and sp1.is_sparse and sp2.is_sparse
    m = sp1.size(0)
    k = sp2.size(0)
    n = sp2.size(-1)
    indices, values = spspmm(
        sp1.indices(), sp1.values(),
        sp2.indices(), sp2.values(),
        m, k, n
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        torch.Size([m, n])
    )
