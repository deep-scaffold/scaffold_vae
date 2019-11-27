import os.path as op
from multiprocessing import cpu_count
import typing as t
from threading import Thread
import abc

# from joblib import Parallel, delayed
import multiprocess as mp
# import multiprocessing as mp
import numpy as np

from .utils import *
from .graph import *

__all__ = [
    'ComLoader',
    'FormalLoader',
]


# def _worker(
#     queue_in,
#     queue_out,
# ):
#     while True:
#         if queue_in.empty():
#             break
#         _block = queue_in.get()

#         nums_nodes = []
#         nums_edges = []
#         bond_infos = []
#         nodes_os = []
#         nodes_cs = []
#         for smiles_i in _block:
#             try:
#                 weave_mol = WeaveMol(smiles_i)
#                 num_nodes = weave_mol.num_all_nodes
#                 num_edges = weave_mol.num_all_edges
#                 all_edges = weave_mol.all_edges
#                 all_nodes_o = weave_mol.all_nodes_o
#                 all_nodes_c = weave_mol.all_nodes_c

#                 nums_nodes.append(num_nodes)
#                 nums_edges.append(num_edges)
#                 bond_infos.append(all_edges)
#                 nodes_os.append(all_nodes_o)
#                 nodes_cs.append(all_nodes_c)
#             except:
#                 pass
#         queue_out.put((
#             _block,
#             nums_nodes,
#             nums_edges,
#             bond_infos,
#             nodes_os,
#             nodes_cs
#         ))


class Loader(object):
    def __init__(
        self,
        original_scaffolds_file: str=op.join(
            op.dirname(__file__),
            '..',
            'data-center',
            'scaffolds_a.smi'
        ),
        # c_scaffolds_file: str=op.join(
        #     op.dirname(__file__),
        #     'data-center',
        #     'scaffolds_c.smi'
        # ),
        batch_size: int=400,
        num_workers: int=cpu_count()
    ):
        super().__init__()
        self.o_scaffolds = original_scaffolds_file
        # self.c_scaffolds = c_scaffolds_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_line = get_num_lines(self.o_scaffolds)
        self.smiles_blocks = str_block_gen(
            self.o_scaffolds,
            self.batch_size
        )
        self.num_id_block = len(self.smiles_blocks)

        # self.num_train_blocks = self.num_id_block // 10 * 8
        # self.num_test_blocks = self.num_id_block - self.num_train_blocks

        # self.train_blocks = self.smiles_blocks[:self.num_train_blocks]
        # self.test_blocks = self.smiles_blocks[self.num_train_blocks:]

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError


class FormalLoader(Loader):
    def __init__(
        self,
        event: mp.synchronize.Event=mp.Event(),
        # worker: t.Callable=_worker,
        **kargs
    ):
        super(FormalLoader, self).__init__(**kargs)
        self.event = event
        self.manager = mp.Manager()
        self.queue_in, self.queue_out = (
            self.manager.Queue(),
            self.manager.Queue(self.num_workers * 3)
        )

        for smiles_block_i in self.smiles_blocks:
            self.queue_in.put(smiles_block_i)

        # self.worker = worker
        self.pool = mp.Pool(self.num_workers)

    def __len__(self):
        return self.num_id_block

    def start_pool(self):
        def _worker(
            block,
            queue_out
        ):
            nums_nodes = []
            nums_edges = []
            bond_infos = []
            nodes_os = []
            nodes_cs = []
            for smiles_i in block:
                try:
                    weave_mol = WeaveMol(smiles_i)
                    num_nodes = weave_mol.num_all_nodes
                    num_edges = weave_mol.num_all_edges
                    all_edges = weave_mol.all_edges
                    all_nodes_o = weave_mol.all_nodes_o
                    all_nodes_c = weave_mol.all_nodes_c

                    nums_nodes.append(num_nodes)
                    nums_edges.append(num_edges)
                    bond_infos.append(all_edges)
                    nodes_os.append(all_nodes_o)
                    nodes_cs.append(all_nodes_c)
                except:
                    pass
            queue_out.put((
                block,
                nums_nodes,
                nums_edges,
                bond_infos,
                nodes_os,
                nodes_cs
            ))
        for block in self.smiles_blocks:
            self.pool.apply(_worker, (block, self.queue_out))

    def _stop_pool(self):
        self.pool.terminate()
        self.pool.close()

    def _worker(self, _block):
        # while True:
        #     if self.queue_in.empty():
        #         break

        #     _block = self.queue_in.get()

        nums_nodes = []
        nums_edges = []
        bond_infos = []
        nodes_os = []
        nodes_cs = []
        for smiles_i in _block:
            try:
                weave_mol = WeaveMol(smiles_i)
                num_nodes = weave_mol.num_all_nodes
                num_edges = weave_mol.num_all_edges
                all_edges = weave_mol.all_edges
                all_nodes_o = weave_mol.all_nodes_o
                all_nodes_c = weave_mol.all_nodes_c

                nums_nodes.append(num_nodes)
                nums_edges.append(num_edges)
                bond_infos.append(all_edges)
                nodes_os.append(all_nodes_o)
                nodes_cs.append(all_nodes_c)
            except:
                pass
        self.queue_out.put((
            _block,
            nums_nodes,
            nums_edges,
            bond_infos,
            nodes_os,
            nodes_cs
        ))

    def __iter__(self):
        for i in range(self.num_id_block):
            if i == 0:
                self._start_pool()
            if self.event.is_set():
                self._stop_pool()
                raise StopIteration
            (
                block,
                nums_nodes,
                nums_edges,
                bond_infos,
                nodes_os,
                nodes_cs
            ) = self.queue_out.get()
            seg_ids = np.arange(len(nums_nodes)).repeat(nums_nodes)

            bond_info = np.concatenate(bond_infos, axis=0)

            shift_values = np.pad(
                np.cumsum(nums_nodes),
                1,
                'constant',
                constant_values=0
            )[: -2].repeat(nums_edges).reshape([-1, 1])

            bond_info_all = bond_info + shift_values

            nodes_o = np.concatenate(nodes_os, axis=-1)
            nodes_c = np.concatenate(nodes_cs, axis=-1)

            yield (
                block,
                nums_nodes,
                nums_edges,
                seg_ids,
                bond_info_all,
                nodes_o,
                nodes_c
            )


class ComLoader(Loader):
    def __init__(self, **kargs):
        super(ComLoader, self).__init__(**kargs)

    def __len__(self):
        return self.num_id_block

    def __iter__(self):
        queue_in, queue_out = (
            mp.Queue(self.num_workers * 3),
            mp.Queue(self.num_workers * 3)
        )

        def _worker_g():
            for smiles_block_i in self.smiles_blocks:
                queue_in.put(smiles_block_i)
            for _ in range(self.num_workers):
                queue_in.put(None)

        def _workder():
            while True:
                _block = queue_in.get()
                if _block is None:
                    break
                nums_nodes = []
                nums_edges = []
                bond_infos = []
                nodes_os = []
                nodes_cs = []
                for smiles_i in _block:
                    try:
                        weave_mol = WeaveMol(smiles_i)
                        num_nodes = weave_mol.num_all_nodes
                        num_edges = weave_mol.num_all_edges
                        all_edges = weave_mol.all_edges
                        all_nodes_o = weave_mol.all_nodes_o
                        all_nodes_c = weave_mol.all_nodes_c

                        nums_nodes.append(num_nodes)
                        nums_edges.append(num_edges)
                        bond_infos.append(all_edges)
                        nodes_os.append(all_nodes_o)
                        nodes_cs.append(all_nodes_c)
                    except:
                        pass
                queue_out.put((
                    _block,
                    nums_nodes,
                    nums_edges,
                    bond_infos,
                    nodes_os,
                    nodes_cs
                ))
            queue_out.put(None)

        thread = Thread(target=_worker_g)
        thread.start()

        pool = [mp.Process(target=_workder) for _ in range(self.num_workers)]
        for p in pool:
            p.start()

        exit_workers = 0
        while exit_workers < self.num_workers:
            record = queue_out.get()
            if record is None:
                exit_workers += 1
                continue
            (
                block,
                nums_nodes,
                nums_edges,
                bond_infos,
                nodes_os,
                nodes_cs
            ) = record

            seg_ids = np.arange(len(nums_nodes)).repeat(nums_nodes)

            bond_info = np.concatenate(bond_infos, axis=0)

            shift_values = np.pad(
                np.cumsum(nums_nodes),
                1,
                'constant',
                constant_values=0
            )[: -2].repeat(nums_edges).reshape([-1, 1])

            bond_info_all = bond_info + shift_values

            nodes_o = np.concatenate(nodes_os, axis=-1)
            nodes_c = np.concatenate(nodes_cs, axis=-1)

            yield (
                block,
                nums_nodes,
                nums_edges,
                seg_ids,
                bond_info_all,
                nodes_o,
                nodes_c
            )

