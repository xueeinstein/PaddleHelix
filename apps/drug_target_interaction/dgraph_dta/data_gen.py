#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
data gen
"""

import random
import numpy as np
from glob import glob

from paddle import fluid
import pgl
from pgl.utils.data.dataset import Dataset, StreamDataset
from pahelix.utils.protein_tools import ProteinTokenizer
from pahelix.utils.data_utils import load_npz_to_data_list


class DTADataset(StreamDataset):
    """DTADataset a subclass of StreamDataset for PGL inputs.
    """
    def __init__(self, data_dir, trainer_id=0, trainer_num=1, subset_selector=None):
        self.subset_selector = subset_selector
        self.cached_len = None
        files = glob('%s/*_%s.npz' % (data_dir, trainer_id))
        files = sorted(files)
        self.files = []
        for (i, f) in enumerate(files):
            if i % trainer_num == trainer_id:
                self.files.append(f)

    def __iter__(self):
        random.shuffle(self.files)
        for f in self.files:
            data_list = load_npz_to_data_list(f)
            if self.subset_selector is not None:
                data_list = self.subset_selector(data_list)
            for data in data_list:
                yield data

    def __len__(self):
        if self.cached_len is not None:
            return self.cached_len
        else:
            n = 0
            for f in self.files:
                data_list = load_npz_to_data_list(f)
                n += len(data_list)

            self.cached_len = n
            return n


class DTACollateFunc(object):
    def __init__(self, compound_graph_wrapper, protein_graph_wrapper,
                 label_name='Log10_Kd', is_inference=False):
        """Collate function for PGL dataloader.

        Args:
            compound_graph_wrapper (pgl.graph_wrapper.GraphWrapper): compound graph wrapper for GNN.
            protein_graph_wrapper (pgl.graph_wrapper.GraphWrapper): protein graph wrapper for GNN.
            label_name (str): the key in the feed dictionary for the drug-target affinity.
                For Davis, it is `Log10_Kd`; For Kiba, it is `KIBA`.
            is_inference (bool): when its value is True, there is no label in the generated feed dictionary.

        Return:
            collate_fn: a callable function.
        """
        assert label_name in ['Log10_Kd', 'Log10_Ki', 'KIBA']
        super(DTACollateFunc, self).__init__()
        self.compound_graph_wrapper = compound_graph_wrapper
        self.protein_graph_wrapper = protein_graph_wrapper
        self.is_inference = is_inference
        self.label_name = label_name

    def __call__(self, batch_data_list):
        """
        Function caller to convert a batch of data into a big batch feed dictionary.

        Args:
            batch_data_list: a batch of the compound graph data and protein sequence tokens data.

        Returns:
            feed_dict: a dictionary contains `graph/xxx` inputs for PGL and `protein_xxx` for protein model.
        """
        compound_graph_list, protein_graph_list = [], []
        for data in batch_data_list:
            atom_numeric_feat = np.concatenate([
                data['atom_degrees'],
                data['atom_Hs'],
                data['atom_implicit_valence'],
                data['atom_is_aromatic'].reshape([-1, 1])
            ], axis=1).astype(np.float32)
            g_compound = pgl.graph.Graph(
                num_nodes = len(data['atom_type']),
                edges = data['edges'],
                node_feat = {
                    'atom_type': data['atom_type'].reshape([-1, 1]),
                    'chirality_tag': data['chirality_tag'].reshape([-1, 1]),
                    'atom_numeric_feat': atom_numeric_feat
                },
                edge_feat = {
                    'bond_type': data['bond_type'].reshape([-1, 1]),
                    'bond_direction': data['bond_direction'].reshape([-1, 1])
                })
            compound_graph_list.append(g_compound)

            g_protein = pgl.graph.Graph(
                num_nodes = len(data['protein_seq_feat']),
                edges = data['protein_edges'],
                node_feat = {
                    'protein_seq_feat': data['protein_seq_feat'].astype(
                        np.float32)
                })
            protein_graph_list.append(g_protein)

        compound_join_graph = pgl.graph.MultiGraph(compound_graph_list)
        protein_join_graph = pgl.graph.MultiGraph(protein_graph_list)
        fd_1 = self.compound_graph_wrapper.to_feed(compound_join_graph)
        fd_2 = self.protein_graph_wrapper.to_feed(protein_join_graph)
        feed_dict = {k: v for k, v in fd_1.items()}
        for k, v in fd_2.items():
            feed_dict[k] = v

        if not self.is_inference:
            batch_label = np.array([data[self.label_name] for data in batch_data_list]).reshape(-1, 1)
            batch_label = batch_label.astype('float32')
            feed_dict['label'] = batch_label
        return feed_dict
