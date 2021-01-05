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

import numpy as np
import pgl
from pgl import graph

from pahelix.utils.compound_tools import CompoundConstants


class MoleculeCollateFunc(object):
    """
    Collate function for molecule dataloader.
    """
    def __init__(self,
                 graph_wrapper,
                 with_graph_label=True,
                 with_attr_mask=False,
                 label_dim=1,  # 1 for reg task, 2 for binary cls
                 mask_ratio=0.15):
        self.graph_wrapper = graph_wrapper
        self.with_graph_label = with_graph_label
        self.with_attr_mask = with_attr_mask
        self.label_dim = label_dim
        self.mask_ratio = mask_ratio

    def __call__(self, batch_data_list):
        """
        Function caller to convert a batch of data into a big batch feed dictionary.

        Args:
            batch_data_list: a batch of the compound graph data.
        """
        g_list, label_list = [], []
        for data in batch_data_list:
            # NOTE: insert a dummy node at the head
            atom_type = np.zeros(data['atom_type'].shape[0] + 1).astype('int')
            atom_type[1:] = data['atom_type']
            g = graph.Graph(
                num_nodes=len(atom_type),
                edges=data['edges'] + 1,
                node_feat={
                    'atom_type': atom_type.reshape([-1, 1])
                })
            g_list.append(g)

        join_graph = pgl.graph.MultiGraph(g_list)

        if self.with_attr_mask:
            # Mask random-selected atom types
            num_node = len(join_graph.node_feat['atom_type'])
            masked_size = int(num_node * self.mask_ratio)
            masked_node_indice = np.random.choice(range(1, num_node), size=masked_size)
            masked_node_labels = join_graph.node_feat['atom_type'][masked_node_indice]
            join_graph.node_feat['atom_type'][masked_node_indice] = len(CompoundConstants.atom_num_list)

        feed_dict = self.graph_wrapper.to_feed(join_graph)

        d_matrix = max([data['matrix_d'] for data in batch_data_list])
        feed_dict['d_matrix'] = np.array([d_matrix], dtype=np.int32)

        adj_matrix = [data['adj_matrix'] for data in batch_data_list]
        adj_matrix_len = [0] + [data['adj_matrix'].size for data in batch_data_list]
        feed_dict['adj_matrix'] = np.concatenate(adj_matrix).reshape([-1, 1]).astype('float32')
        feed_dict['adj_matrix_lod'] = np.add.accumulate(
            adj_matrix_len).reshape([1, -1]).astype('int32')

        dist_matrix = [data['dist_matrix'] for data in batch_data_list]
        dist_matrix_len = [0] + [data['dist_matrix'].size for data in batch_data_list]
        feed_dict['dist_matrix'] = np.concatenate(dist_matrix).reshape([-1, 1]).astype('float32')
        feed_dict['dist_matrix_lod'] = np.add.accumulate(
            dist_matrix_len).reshape([1, -1]).astype('int32')

        feed_dict['mask'], feed_dict['mask_lod'] = self._get_mask(batch_data_list)

        if self.with_graph_label:
            label_list = [data['label'] for data in batch_data_list]
            if label_list[0].shape[-1] != self.label_dim:
                # This is a classification task, use one-hot label
                label_list = [self._to_one_hot(l[0]) for l in label_list]

            feed_dict['label'] = np.array(label_list).astype('float32')

        if self.with_attr_mask:
            # Input for attribute mask self-supervised pretraining
            feed_dict['masked_node_indice'] = np.reshape(
                masked_node_indice, [-1, 1]).astype('int64')
            feed_dict['masked_node_label'] = np.reshape(
                masked_node_labels, [-1, 1]).astype('int64')

        return feed_dict

    def _to_one_hot(self, label):
        one_hot = np.zeros(self.label_dim)
        one_hot[int(label)] = 1.0
        return one_hot

    def _get_mask(self, batch_data_list):
        mask = []
        for data in batch_data_list:
            m = np.ones(int(data['adj_matrix'].size ** 0.5)).astype('float32')
            mask.append(m)

        mask_len = [0] + [m.size for m in mask]
        mask_lod = np.add.accumulate(mask_len).reshape([1, -1]).astype('int32')
        mask = np.concatenate(mask).reshape([-1, 1])
        return mask, mask_lod
