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
DTA model
"""

from paddle import fluid
import pgl
from pgl.graph_wrapper import GraphWrapper

from pahelix.utils.protein_tools import ProteinConstants
from pahelix.utils.compound_tools import CompoundConstants


class DoubleGCN(object):
    """
    | DoubleGCN, implementation of DGraphDTA which encodes both features
        of compound and protein using GNN models.
    """

    def __init__(self, model_config, without_msa=False, name=''):
        self.name = name

        self.compound_embed_dim = model_config['compound_embed_dim']
        self.compound_hidden_size = model_config['compound_hidden_size']
        self.protein_hidden_size = model_config['protein_hidden_size']
        self.dropout_rate = model_config['dropout_rate']
        self.output_dim = model_config['output_dim']
        self.layer_num = model_config['layer_num']

        self.fc_hidden_size = model_config.get('fc_hidden_size', 1024)
        self.atom_type_num = model_config.get(
                'atom_type_num', len(CompoundConstants.atom_num_list) + 2)

        dim = CompoundConstants.atomic_numeric_feat_dim
        self.compound_graph_wrapper = GraphWrapper(
            name='compound_graph',
            node_feat=[
                ('atom_type', [None, 1], "int64"),
                ('chirality_tag', [None, 1], "int64"),
                ('atom_numeric_feat', [None, dim], "float32")],
            edge_feat=[
                ('bond_type', [None, 1], "int64"),
                ('bond_direction', [None, 1], "int64")
            ])

        # PSSM + amino acid type + amino acid properties
        factor = 1 if without_msa else 2
        dim = len(ProteinConstants.amino_acids) * factor + \
            ProteinConstants.amino_acids_properties_dim
        self.protein_graph_wrapper = GraphWrapper(
            name='protein_graph',
            node_feat=[('protein_seq_feat', [None, dim], "float32")])

    def _atom_encoder(self, graph_wrapper, name=""):
        embed_init = fluid.initializer.XavierInitializer(uniform=True)

        atom_type_embed = fluid.layers.embedding(
                input=graph_wrapper.node_feat['atom_type'],
                size=[self.atom_type_num, self.compound_embed_dim],
                param_attr=fluid.ParamAttr(
                    name="%s_embed_atom_type" % name, initializer=embed_init))
        node_features = fluid.layers.concat(
            [atom_type_embed, graph_wrapper.node_feat['atom_numeric_feat']], axis=1)
        return node_features

    def _compound_encoder(self):
        node_features = self._atom_encoder(
            self.compound_graph_wrapper, name=self.name + '_compound')

        features_list = [node_features]
        for layer in range(self.layer_num):
            feat = pgl.layers.gcn(
                self.compound_graph_wrapper,
                features_list[layer],
                self.compound_hidden_size * (2 ** layer),  # 1, 2, 4
                'relu',
                '%s_compound_layer%s' % (self.name, layer))
            features_list.append(feat)

        feat = pgl.layers.graph_pooling(
            self.compound_graph_wrapper, features_list[-1], 'average')

        feat = fluid.layers.fc(feat, self.fc_hidden_size, act='relu',
                               name='%s_compound_fc_1' % self.name)
        feat = fluid.layers.dropout(
            feat, self.dropout_rate, dropout_implementation="upscale_in_train")

        feat = fluid.layers.fc(feat, self.output_dim,
                               name='%s_compound_fc_2' % self.name)
        feat = fluid.layers.dropout(
            feat, self.dropout_rate, dropout_implementation="upscale_in_train")
        return feat

    def _protein_encoder(self):
        node_features = self.protein_graph_wrapper.node_feat['protein_seq_feat']

        features_list = [node_features]
        for layer in range(self.layer_num):
            feat = pgl.layers.gcn(
                self.protein_graph_wrapper,
                features_list[layer],
                self.protein_hidden_size * (2 ** layer),  # 1, 2, 4
                'relu',
                '%s_protein_layer%s' % (self.name, layer))
            features_list.append(feat)

        feat = pgl.layers.graph_pooling(
            self.protein_graph_wrapper, features_list[-1], 'average')

        feat = fluid.layers.fc(feat, self.fc_hidden_size, act='relu',
                               name='%s_protein_fc_1' % self.name)
        feat = fluid.layers.dropout(
            feat, self.dropout_rate, dropout_implementation="upscale_in_train")

        feat = fluid.layers.fc(feat, self.output_dim,
                               name='%s_protein_fc_2' % self.name)
        feat = fluid.layers.dropout(
            feat, self.dropout_rate, dropout_implementation="upscale_in_train")
        return feat

    def forward(self):
        compound_repr = self._compound_encoder()
        protein_repr = self._protein_encoder()

        compound_protein = fluid.layers.concat(
            [compound_repr, protein_repr], axis=1)

        h = fluid.layers.fc(compound_protein, 1024, act='relu')
        h = fluid.layers.dropout(
            h, self.dropout_rate, dropout_implementation='upscale_in_train')

        h = fluid.layers.fc(h, 512, act='relu')
        h = fluid.layers.dropout(
            h, self.dropout_rate, dropout_implementation='upscale_in_train')

        pred = fluid.layers.fc(h, 1)
        return pred

    def train(self):
        label = fluid.layers.data(name="label", dtype='float32', shape=[None, 1])
        pred = self.forward()
        loss = fluid.layers.square_error_cost(input=pred, label=label)
        loss = fluid.layers.mean(loss)

        self.pred = pred
        self.loss = loss

    def inference(self):
        self.pred = self.forward()
