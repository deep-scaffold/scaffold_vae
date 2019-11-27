import typing as t
from copy import deepcopy
from itertools import chain

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdmolops
from scipy import sparse

from mol_spec import MoleculeSpec


__all__ = [
    # "smiles_to_nx_graph",
    "WeaveMol",
]

ms = MoleculeSpec.get_default()


class WeaveMol(object):
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.ms = ms
        self.num_atom_types = ms.num_atom_types
        self.num_bond_types = ms.num_bond_types
        self.num_atoms = self.mol.GetNumAtoms()
        self.original_atoms = list(range(self.num_atoms))
        self.num_original_bonds = self.mol.GetNumBonds()
        self._atom_types = None
        self._original_bond_info = None
        self._original_bond_info_np = None
        self._original_bond_types = None
        self._original_graph = None
        self._original_adj = None
        self._remote_connections = None
        self._sssr = None
        self._new_bond_info = None
        self._new_bond_info_np = None
        self._new_sssr = None
        self._new_ring_atoms = None
        self._new_graph = None
        self._new_chain_nodes = None
        self._ring_assems = None
        self._chain_assems = None
        self._remote_connection_bonds = None
        self._chains_master_nodes = None
        self._chains_master_edges = None
        self._rings_master_nodes = None
        self._rings_master_edges = None
        self._ringassems_master_nodes = None
        self._ringassems_master_edges = None
        self._mol_master_nodes = None
        self._mol_master_edges = None
        self._all_nodes_o = None
        self._all_nodes_c = None
        self._all_edges = None
        self._node_ringassesm_idx_dic = None

    @property
    def atom_types(self):
        if self._atom_types is not None:
            return self._atom_types
        else:
            atom_types = [
                ms.get_atom_type(atom) for atom in self.mol.GetAtoms()
            ]
            self._atom_types = np.array(atom_types, dtype=np.int)
            return self._atom_types

    @property
    def atom_type_c(self):
        return np.zeros_like(self.atom_types, dtype=np.int)

    @property
    def original_bond_info(self):
        if self._original_bond_info is not None:
            return self._original_bond_info
        else:
            self._original_bond_info = [
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                for bond in self.mol.GetBonds()
            ]
        return self._original_bond_info  # N_bond x 2

    @property
    def original_bond_types(self, to_array=False):
        if self._original_bond_types is not None:
            return self._original_bond_types
        else:
            bond_types = [
                ms.get_bond_type(bond) for bond in self.mol.GetBonds()
            ]
            self._original_bond_types = np.array(bond_types, dtype=np.int)
        return self._original_bond_types

    @property
    def original_bond_types_c(self):
        return np.zeros_like(self.original_bond_types, dtype=np.int)

    @property
    def original_graph(self) -> nx.classes.graph.Graph:
        if self._original_graph is not None:
            return self._original_graph
        else:
            graph = nx.Graph()
            graph.add_nodes_from(range(self.num_atoms))
            graph.add_edges_from(self.original_bond_info)
            self._original_graph = graph
        return self._original_graph

    @property
    def original_adj(self):
        if self._original_adj is not None:
            return self._original_adj
        else:
            indices1 = self.original_bond_info_np  # N_bonds x 2
            indices2 = np.flip(indices1, axis=-1)
            # indices = np.concatenate([indices1, indices2], axis=0)
            indices = np.stack(
                [indices1, indices2], axis=1
            ).transpose(0, 2, 1).reshape(-1, 2)
            row, col, data = (
                indices[:, 0],
                indices[:, 1],
                np.ones([indices.shape[0], ], dtype=np.float32)
            )
            self._original_adj = sparse.coo_matrix(
                (data, (row, col)),
                shape=(self.num_atoms, self.num_atoms)
            )
        return self._original_adj

    @property
    def original_bond_info_np(self):
        if self._original_bond_info_np is not None:
            return self._original_bond_info_np
        else:
            self._original_bond_info_np = np.stack(self.original_bond_info)
            return self._original_bond_info_np

    @property
    def remote_connections(self):
        if self._remote_connections is not None:
            return self._remote_connections
        else:
            d = self.original_adj * self.original_adj
            d_indices_2 = np.stack(d.nonzero(), axis=1)
            # remove diagonal elements
            d_indices_2 = d_indices_2[
                d_indices_2[:, 0] != d_indices_2[:, 1], :
            ]

            d = d * self.original_adj
            d = d - d.multiply(self.original_adj)

            d_indices_3 = np.stack(d.nonzero(), axis=1)
            # remove diagonal elements
            d_indices_3 = d_indices_3[
                d_indices_3[:, 0] != d_indices_3[:, 1], :
            ]

            self._remote_connections = (
                d_indices_2[d_indices_2[:, 0] < d_indices_2[:, 1]],
                d_indices_3[d_indices_3[:, 0] < d_indices_3[:, 1]]
            )
            return self._remote_connections

    @property
    def remote_connection_2(self):
        return self.remote_connections[0]

    @property
    def remote_connection_3(self):
        return self.remote_connections[1]

    @property
    def num_remote_connection_2(self):
        return self.remote_connections[0].shape[0]

    @property
    def num_remote_connection_3(self):
        return self.remote_connections[1].shape[0]

    @property
    def sssr(self) -> t.List[t.List]:
        """Get sssr atom indices of a molecule
        
        Returns:
            t.List[t.List]: [[sssr1 atomindices],[sssr2 atom indices], ...]
        """
        if self._sssr is not None:
            return self._sssr
        else:
            self._sssr = [
                list(ring) for ring in rdmolops.GetSymmSSSR(self.mol)
            ]
            return self._sssr

    @property
    def num_original_rings(self) -> int:
        return len(self.sssr)

    @property
    def new_atoms(self):
        return (
            list(
                range(
                    self.num_atoms,
                    self.num_atoms + self.num_original_bonds
                )
            )
        )

    @property
    def new_nodes(self):
        return self.original_atoms + self.new_atoms

    @property
    def new_bond_info(self):
        if self._new_bond_info is not None:
            return self._new_bond_info
        else:
            new_bond_info = np.concatenate([
                np.stack([
                    self.original_bond_info_np[:, 0],
                    self.new_atoms
                ], axis=0),
                np.stack([
                    self.new_atoms,
                    self.original_bond_info_np[:, 1]
                ], axis=0)
            ], axis=-1).T

            self._new_bond_info = new_bond_info
            return self._new_bond_info

    #  here to cmodify
    @property
    def new_bond_info_np(self):
        if self._new_bond_info_np is not None:
            return self._new_bond_info_np
        else:
            self._new_bond_info_np = np.concatenate(
                [
                    self.original_bond_info_np,
                    np.array(
                        self.new_atoms,
                        dtype=np.int
                    ).reshape([-1, 1])
                ],
                axis=1
            )
        return self._new_bond_info_np

    @property
    def new_bond_info_dict(self):
        new_bond_info_dict = {}
        for new_bond in self.new_bond_info_np:
            new_bond_info_dict[(new_bond[0], new_bond[1])] = new_bond[2]
        return new_bond_info_dict

    # @property
    # def new_edge_info(self):
    #     bonds1 = np.concatenate(
    #         [
    #             self.new_bond_info_np[:, 0].reshape([-1, 1]),
    #             self.new_bond_info_np[:, 2].reshape([-1, 1])
    #         ], axis=-1
    #     )
    #     bonds2 = np.concatenate(
    #         [
    #             self.new_bond_info_np[:, 2].reshape([-1, 1]),
    #             self.new_bond_info_np[:, 1].reshape([-1, 1])
    #         ], axis=-1
    #     )
    #     bonds = np.concatenate([bonds1, bonds2], axis=0)
    #     return bonds

    @property
    def original_ring_bond_info(self):
        ring_bond_info = []
        ring_bond_info_set = []
        for ring in self.sssr:
            ring_bond_set = []
            num_ring_atoms = len(ring)
            for idx_atom in range(num_ring_atoms):
                if idx_atom >= num_ring_atoms - 1:
                    if (ring[idx_atom], ring[0]) in self.original_bond_info:
                        bond_info = (ring[idx_atom], ring[0])
                    else:
                        bond_info = (ring[0], ring[idx_atom])
                    ring_bond_info.append(bond_info)
                    ring_bond_set.append(bond_info)
                else:
                    if (
                        (ring[idx_atom], ring[idx_atom + 1])
                        in self.original_bond_info
                    ):
                        bond_info = (ring[idx_atom], ring[idx_atom + 1])
                    else:
                        bond_info = (ring[idx_atom + 1], ring[idx_atom])
                    ring_bond_info.append(bond_info)
                    ring_bond_set.append(bond_info)
            ring_bond_info_set.append(ring_bond_set)
        return list(set(ring_bond_info)), ring_bond_info_set

    @property
    def original_chain_bond_info(self):
        return (
            list(
                set(self.original_bond_info) -
                set(self.original_ring_bond_info[0])
            )
        )

    @property
    def new_sssr_legacy(self):
        new_sssr = deepcopy(self.sssr)
        for i, sssr in enumerate(new_sssr):
            for bond_info in self.original_ring_bond_info[1][i]:
                sssr.append(self.new_bond_info_dict[bond_info])
        return new_sssr

    @property
    def new_sssr(self):

        if self._new_sssr is not None:
            return self._new_sssr
        else:
            new_sssr = deepcopy(self.sssr)
            bond_info = np.expand_dims(self.original_bond_info, -1)
            new_atoms = np.array(self.new_atoms, dtype=np.int)
            for ring in new_sssr:
                ring_np = np.array(ring, dtype=np.int)
                in_ring = (bond_info == ring_np).astype(np.int).reshape(
                    [len(self.new_atoms), -1]
                )
                num_in_ring = in_ring.sum(axis=-1)
                where_both_in = np.where(num_in_ring > 1)
                new_atoms_this_ring = new_atoms[where_both_in].tolist()
                ring.extend(new_atoms_this_ring)
            self._new_sssr = new_sssr
            return self._new_sssr

    @property
    def new_ring_atoms(self):
        if self._new_ring_atoms is not None:
            return self._new_ring_atoms
        else:
            self._new_ring_atoms = list(set(chain(*self.new_sssr)))
            return self._new_ring_atoms

    @property
    def new_graph(self):
        if self._new_graph is not None:
            return self._new_graph
        else:
            g = nx.Graph()
            g.add_nodes_from(self.new_nodes)
            g.add_edges_from(self.new_bond_info)
            self._new_graph = g
            return self._new_graph

    @property
    def new_chain_nodes(self):
        if self._new_chain_nodes is not None:
            return self._new_chain_nodes
        else:
            self._new_chain_nodes = (
                list(
                    set(self.new_nodes) - set(self.new_ring_atoms)
                )
            )
            return self._new_chain_nodes

    @property
    def ring_assems(self):
        if self._ring_assems is not None:
            return self._ring_assems
        else:
            g = deepcopy(self.new_graph)
            g.remove_nodes_from(self.new_chain_nodes)
            ring_assems = list(nx.connected_component_subgraphs(g))
            self._ring_assems = [list(graph.nodes()) for graph in ring_assems]
            return self._ring_assems

    @property
    def chain_assems(self):
        if self._chain_assems is None:
            g = deepcopy(self.new_graph)
            g.remove_nodes_from(self.new_ring_atoms)
            chain_assems = list(nx.connected_component_subgraphs(g))
            self._chain_assems = [
                list(graph.nodes()) for graph in chain_assems
            ]
        return self._chain_assems

    @property
    def remote_connection_bonds(self):
        if self._remote_connection_bonds is None:
            num_new_nodes = len(self.new_nodes)
            remote_nodes_2 = list(range(
                num_new_nodes,
                num_new_nodes + self.num_remote_connection_2
            ))
            remote_nodes_3 = list(range(
                num_new_nodes + self.num_remote_connection_2,
                num_new_nodes + self.num_remote_connection_2 +
                self.num_remote_connection_3
            ))
            remote_bonds2 = np.concatenate([
                np.concatenate([
                    self.remote_connection_2[:, 0].reshape([-1, 1]),
                    np.array(remote_nodes_2, dtype=np.int).reshape([-1, 1])
                ], axis=-1),
                np.concatenate([
                    np.array(remote_nodes_2, dtype=np.int).reshape([-1, 1]),
                    self.remote_connection_2[:, 1].reshape([-1, 1])
                ], axis=-1)
            ], axis=0)
            remote_bonds3 = np.concatenate([
                np.concatenate([
                    self.remote_connection_3[:, 0].reshape([-1, 1]),
                    np.array(remote_nodes_3, dtype=np.int).reshape([-1, 1])
                ], axis=-1),
                np.concatenate([
                    np.array(remote_nodes_3, dtype=np.int).reshape([-1, 1]),
                    self.remote_connection_3[:, 1].reshape([-1, 1])
                ], axis=-1)
            ], axis=0)
            self._remote_connection_bonds = (
                remote_bonds2, remote_bonds3
            )
        return self._remote_connection_bonds

    @property
    def remote_bonds_2(self):
        return self.remote_connection_bonds[0]

    @property
    def remote_bonds_3(self):
        return self.remote_connection_bonds[1]

    @property
    def chains_master_nodes(self):
        if self._chains_master_nodes is None:
            num_above = (
                self.num_atoms +
                self.num_original_bonds +
                self.num_remote_connection_2 +
                self.num_remote_connection_3
            )
            self._chains_master_nodes = list(
                range(
                    num_above, num_above + len(self.chain_assems)
                )
            )
        return self._chains_master_nodes

    @property
    def num_chains_master_nodes(self):
        return len(self.chains_master_nodes)

    @property
    def chains_master_edges(self):
        if self._chains_master_edges is None:
            num_atoms_each_chain = [len(chain) for chain in self.chain_assems]
            master_nodes = np.repeat(
                self.chains_master_nodes,
                num_atoms_each_chain,
                axis=0
            )
            all_chain_nodes = list(chain(*self.chain_assems))
            self._chains_master_edges = np.stack(
                [master_nodes, all_chain_nodes], axis=0
            ).T
        return self._chains_master_edges

    @property
    def rings_master_nodes(self):
        if self._rings_master_nodes is None:
            num_above = (
                self.num_atoms +
                self.num_original_bonds +
                self.num_remote_connection_2 +
                self.num_remote_connection_3 +
                self.num_chains_master_nodes
            )
            self._rings_master_nodes = list(
                range(
                    num_above, num_above + self.num_original_rings
                )
            )
        return self._rings_master_nodes

    @property
    def num_rings_master_nodes(self):
        return len(self.rings_master_nodes)

    @property
    def rings_master_edges(self):
        if self._rings_master_edges is None:
            num_atoms_each_ring = [len(ring) for ring in self.new_sssr]
            master_nodes = np.repeat(
                self.rings_master_nodes,
                num_atoms_each_ring,
                axis=0
            )
            all_ring_nodes = list(chain(*self.new_sssr))
            self._rings_master_edges = np.stack(
                [master_nodes, all_ring_nodes], axis=0
            ).T
        return self._rings_master_edges

    @property
    def ringassems_master_nodes(self):
        if self._ringassems_master_nodes is None:
            num_above = (
                self.num_atoms +
                self.num_original_bonds +
                self.num_remote_connection_2 +
                self.num_remote_connection_3 +
                self.num_chains_master_nodes +
                self.num_original_rings
            )
            self._ringassesm_master_nodes = list(
                range(
                    num_above, num_above + self.num_ringassems_master_nodes
                )
            )
        return self._ringassesm_master_nodes

    @property
    def num_ringassems_master_nodes(self):
        return len(self.ring_assems)

    @property
    def node_ringassesm_idx_dic(self):
        if self._node_ringassesm_idx_dic is None:
            self._node_ringassesm_idx_dic = {}
            for i, nodes in enumerate(self.ring_assems):
                ring_dic = dict.fromkeys(nodes, i)
                self._node_ringassesm_idx_dic.update(ring_dic)
        return self._node_ringassesm_idx_dic

    @property
    def ringassems_master_edges(self):
        if self._ringassems_master_edges is None:
            ring_belong_to_ring_assems_idx = [
                self.node_ringassesm_idx_dic[ring[0]] for ring in self.new_sssr
            ]
            master_nodes = np.array(self.ringassems_master_nodes, dtype=np.int)
            master_nodes = master_nodes[ring_belong_to_ring_assems_idx]
            self._ringassems_master_edges = np.stack(
                [master_nodes, self.rings_master_nodes], axis=0
            ).T
        return self._ringassems_master_edges

    @property
    def mol_master_nodes(self):
        if self._mol_master_nodes is None:
            num_above = (
                self.num_atoms +
                self.num_original_bonds +
                self.num_remote_connection_2 +
                self.num_remote_connection_3 +
                self.num_chains_master_nodes +
                self.num_original_rings +
                self.num_ringassems_master_nodes
            )
            self._mol_master_nodes = [num_above]
        return self._mol_master_nodes

    @property
    def mol_master_edges(self):
        if self._mol_master_edges is None:
            sub_level_nodes = (
                self.chains_master_nodes + self.ringassems_master_nodes
            )
            master_nodes = np.repeat(
                self.mol_master_nodes[0],
                len(sub_level_nodes)
            )
            self._mol_master_edges = np.stack(
                [master_nodes, sub_level_nodes], axis=0
            ).T
        return self._mol_master_edges

    @property
    def all_nodes_o(self):
        if self._all_nodes_o is None:
            self._all_nodes = np.concatenate(
                [
                    self.atom_types,

                    self.original_bond_types + self.num_atom_types,

                    np.zeros(self.num_remote_connection_2, dtype=np.int) +
                    self.num_atom_types + self.num_bond_types,

                    np.zeros(self.num_remote_connection_3, dtype=np.int) +
                    self.num_atom_types + self.num_bond_types + 1,

                    np.zeros(self.num_chains_master_nodes, dtype=np.int) +
                    self.num_atom_types + self.num_bond_types + 2,

                    np.zeros(self.num_rings_master_nodes, dtype=np.int) +
                    self.num_atom_types + self.num_bond_types + 3,

                    np.zeros(self.num_ringassems_master_nodes, dtype=np.int) +
                    self.num_atom_types + self.num_bond_types + 4,

                    np.zeros(1, dtype=np.int) +
                    self.num_atom_types + self.num_bond_types + 5
                ], axis=-1
            )
        return self._all_nodes

    @property
    def all_nodes_c(self):
        if self._all_nodes_c is None:
            self._all_nodes_c = np.concatenate(
                [
                    self.atom_type_c,

                    self.original_bond_types_c + 1,

                    np.zeros(self.num_remote_connection_2, dtype=np.int) + 2,

                    np.zeros(self.num_remote_connection_3, dtype=np.int) + 3,

                    np.zeros(self.num_chains_master_nodes, dtype=np.int) + 4,

                    np.zeros(self.num_rings_master_nodes, dtype=np.int) + 5,

                    np.zeros(self.num_ringassems_master_nodes, dtype=np.int) +
                    6,

                    np.zeros(1, dtype=np.int) + 7
                ], axis=-1
            )
        return self._all_nodes_c

    @property
    def all_edges(self):
        if self._all_edges is None:
            all_edges_order = np.concatenate(
                [
                    self.new_bond_info,
                    self.remote_bonds_2,
                    self.remote_bonds_3,
                    self.rings_master_edges,
                    self.chains_master_edges,
                    self.ringassems_master_edges,
                    self.mol_master_edges
                ], axis=0
            )
            all_edges_flipped = np.flip(all_edges_order, axis=-1)
            self._all_edges = np.concatenate(
                [all_edges_order, all_edges_flipped],
                axis=0
            )
        return self._all_edges

    @property
    def num_all_nodes(self):
        return len(self.all_nodes_o)

    @property
    def num_all_edges(self):
        return self.all_edges.shape[0]
