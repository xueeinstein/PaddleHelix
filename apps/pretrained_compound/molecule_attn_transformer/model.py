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
Model for Molecule Attention Transformer
"""

import numpy as np
from paddle import fluid
import pgl
from pgl.graph_wrapper import GraphWrapper

from pahelix.utils.compound_tools import CompoundConstants
from pahelix.networks.transformer_block import positionwise_feed_forward


def apply_dist_matrix_kernel(kernel, x):
    if kernel == 'softmax':
        return fluid.layers.softmax(x, use_cudnn=True)
    elif kernel == 'exp':
        return fluid.layers.exp(-x)
    else:
        raise ValueError('Unknown kernel')


def multi_head_mol_attention(queries,
                             keys,
                             values,
                             mask,
                             d_key,
                             d_value,
                             d_model,
                             adj_matrix,
                             dist_matrix,
                             d_matrix,
                             n_head=1,
                             eps=1e-6,
                             inf=1e10,
                             dropout_rate=0.,
                             lambdas=(0.3, 0.3, 0.4),
                             trainable_lambda=False,
                             param_initializer=None,
                             dist_matrix_kernel=None,
                             name='multi_head_mol_attn'):
    """
    Multi-head molecule attention.

    A_i = (lambda_attn * sigma(QK^T/\sqrt{d}) + lambda_dist * g(D) + lambda_adj * A) * V_i
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        q = fluid.layers.fc(input=queries,
                            size=d_key * n_head,
                            num_flatten_dims=2,
                            param_attr=fluid.ParamAttr(
                                name=name + "_query_fc.w_0",
                                initializer=param_initializer),
                            bias_attr=name + "_query_fc.b_0")
        k = fluid.layers.fc(input=keys,
                            size=d_key * n_head,
                            num_flatten_dims=2,
                            param_attr=fluid.ParamAttr(
                                name=name + "_key_fc.w_0",
                                initializer=param_initializer),
                            bias_attr=name + "_key_fc.b_0")
        v = fluid.layers.fc(input=values,
                            size=d_value * n_head,
                            num_flatten_dims=2,
                            param_attr=fluid.ParamAttr(
                                name=name + "_value_fc.w_0",
                                initializer=param_initializer),
                            bias_attr=name + "_value_fc.b_0")
        return q, k, v

    def __split_heads(x, n_head):
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = fluid.layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return fluid.layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = fluid.layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return fluid.layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    # Pad lod-tensor to [bs, max_sequence_len, emb_dim]
    # pad_value = fluid.layers.assign(input=np.array([0.0], dtype=np.float32))
    # queries, queries_len = fluid.layers.sequence_pad(queries, pad_value=pad_value)
    # keys, keys_len = fluid.layers.sequence_pad(keys, pad_value=pad_value)
    # values, values_len = fluid.layers.sequence_pad(values, pad_value=pad_value)

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)
    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    adj_matrix = adj_matrix / \
        (fluid.layers.reduce_sum(adj_matrix, dim=-1, keep_dim=True) + eps)
    adj_matrix = fluid.layers.reshape(adj_matrix, [-1, 1, d_matrix, d_matrix])

    if dist_matrix_kernel:
        dist_matrix = apply_dist_matrix_kernel(dist_matrix_kernel, dist_matrix)
    dist_matrix = fluid.layers.reshape(dist_matrix, [-1, 1, d_matrix, d_matrix])

    # [bs, n_head, d_matrix, d_matrix],
    # in the input, we guarantee `d_matrix` = `max_sequence_len`
    p_adj = fluid.layers.expand(adj_matrix, [1, n_head, 1, 1])
    p_dist = fluid.layers.expand(dist_matrix, [1, n_head, 1, 1])

    # [bs, n_head, max_sequence_len, max_sequence_len]
    scores = fluid.layers.matmul(x=q, y=k, transpose_y=True) / np.sqrt(d_key)
    if mask:
        # mask = fluid.layers.Print(mask, message="mask: ")
        # d_matrix = fluid.layers.Print(d_matrix, message="d_matrix: ")
        mask = fluid.layers.reshape(mask, [-1, 1, 1, d_matrix])
        mask = fluid.layers.expand(mask, [1, n_head, d_matrix, 1])
        scores = scores * mask - inf * (1 - mask)

    p_attn = fluid.layers.softmax(scores, use_cudnn=True)
    if dropout_rate:
        p_attn = fluid.layers.dropout(
            p_attn, dropout_prob=dropout_rate,
            dropout_implementation='upscale_in_train')

    if trainable_lambda:
        default_init = fluid.initializer.ConstantInitializer(value=1/3.)
        lambda_attn = fluid.create_parameter(
            shape=[1], dtype='float32', default_initializer=default_init)
        lambda_dist = fluid.create_parameter(
            shape=[1], dtype='float32', default_initializer=default_init)
        lambda_adj = fluid.create_parameter(
            shape=[1], dtype='float32', default_initializer=default_init)
    else:
        lambda_attn, lambda_dist, lambda_adj = lambdas

    p_weighted = lambda_attn * p_attn + lambda_dist * p_dist + lambda_adj * p_adj

    if dropout_rate:
        p_weighted = fluid.layers.dropout(
            p_weighted, dropout_prob=dropout_rate,
            dropout_implementation='upscale_in_train')

    # [bs, max_sequence_len, d_value * n_head]
    atom_features = fluid.layers.matmul(x=p_weighted, y=v)
    atom_features = __combine_heads(atom_features)

    atom_features = fluid.layers.fc(input=atom_features,
                                    size=d_model,
                                    num_flatten_dims=2,
                                    param_attr=fluid.ParamAttr(
                                        name=name + "_output_fc.w_0",
                                        initializer=param_initializer),
                                    bias_attr=name + "_output_fc.b_0")
    return atom_features, p_weighted, p_attn


class GraphTransformer(object):
    """
    | GraphTransformer, implementation of the molecule attention transformer.
        ``Molecule Attention Transformer``.

    Public Functions:
    """
    def __init__(self, model_config, name=''):
        self.name = name

        self.embed_dim = model_config['embed_dim']
        self.hidden_size = model_config['hidden_size']
        self.head_num = model_config['head_num']
        self.block_num = model_config.get('block_num', 2)
        self.ffn_layer_num = model_config.get('ffn_layer_num', 2)
        self.pred_layer_num = model_config.get('pred_layer_num', 1)
        self.output_dim = model_config.get('output_dim', 1)
        self.dropout_rate = model_config.get('dropout_rate', 0.0)

        self.lambdas = model_config.get('lambdas', [0.3, 0.3, 0.4])
        self.trainable_lambda = model_config.get('trainable_lambda', False)
        self.dist_matrix_kernel = model_config.get('dist_matrix_kernel', 'softmax')

        self.aggregation_type = model_config.get('aggregation_type', 'mean')

    def forward(self, for_attrmask_pretrain=False):
        graph_wrapper = GraphWrapper(
            name='graph',
            node_feat=[('atom_type', [None, 1], 'int64')])
        node_feat = fluid.layers.embedding(
            input=graph_wrapper.node_feat['atom_type'],
            size=[len(CompoundConstants.atom_num_list)+2, self.embed_dim],
            param_attr=fluid.ParamAttr(
                name='%s_embed_atom_type' % self.name,
                initializer=fluid.initializer.XavierInitializer(uniform=True),
                trainable=True))
        node_feat = fluid.layers.lod_reset(
            node_feat, graph_wrapper.graph_lod)

        mask = fluid.layers.data(name='mask', shape=[None, 1], dtype='float32')
        mask_lod = fluid.layers.data(name='mask_lod', shape=[None], dtype='int32')
        mask_input = fluid.layers.lod_reset(mask, y=mask_lod)

        d_matrix = fluid.layers.data(name='d_matrix', shape=[1], dtype='int32')

        adj_matrix = fluid.layers.data(name='adj_matrix', shape=[None, 1], dtype='float32')
        adj_matrix_lod = fluid.layers.data(
            name='adj_matrix_lod', shape=[None], dtype='int32')
        adj_matrix_input = fluid.layers.lod_reset(adj_matrix, y=adj_matrix_lod)

        dist_matrix = fluid.layers.data(
            name='dist_matrix', shape=[None, 1], dtype='float32')
        dist_matrix_lod = fluid.layers.data(
            name='dist_matrix_lod', shape=[None, 1], dtype='int32')
        dist_matrix_input = fluid.layers.lod_reset(dist_matrix, y=dist_matrix_lod)

        pad_value = fluid.layers.assign(input=np.array([0.0], dtype=np.float32))
        node_feat, node_len = fluid.layers.sequence_pad(node_feat, pad_value=pad_value)

        mask, _ = fluid.layers.sequence_pad(mask_input, pad_value=pad_value)

        adj_matrix, _ = fluid.layers.sequence_pad(
            adj_matrix_input, pad_value=pad_value)
        dist_matrix, _ = fluid.layers.sequence_pad(
            dist_matrix_input, pad_value=pad_value)

        atom_features, _, _ = self.encode(
            node_feat, mask, adj_matrix, dist_matrix, d_matrix[0])

        if for_attrmask_pretrain:
            pred = fluid.layers.sequence_unpad(atom_features * mask, node_len)
        else:
            pred = self.predict(atom_features, mask, d_matrix[0])
        return graph_wrapper, pred

    def encode(self, src, src_mask, adj_matrix, dist_matrix, d_matrix):
        pad_value = fluid.layers.assign(input=np.array([0.0], dtype=np.float32))
        features_list, p_weighted_list, p_attn_list = [src], [], []
        for block_id in range(self.block_num):
            x = features_list[block_id]
            d = x.shape[-1]
            atom_features, p_weighted, p_attn = multi_head_mol_attention(
                x, x, x, src_mask, d, d, self.hidden_size,
                adj_matrix, dist_matrix, d_matrix,
                n_head=self.head_num,
                dropout_rate=self.dropout_rate,
                lambdas=self.lambdas,
                trainable_lambda=self.trainable_lambda,
                dist_matrix_kernel=self.dist_matrix_kernel,
                name='block_%d_attn' % block_id)

            for ffn_id in range(self.ffn_layer_num):
                atom_features = positionwise_feed_forward(
                    atom_features, self.hidden_size, self.hidden_size,
                    self.dropout_rate, 'relu',
                    name='block_%d_ffn_%d' % (block_id, ffn_id))

            features_list.append(atom_features)
            p_weighted_list.append(p_weighted)
            p_attn_list.append(p_attn)

        atom_features = fluid.layers.layer_norm(
            features_list[-1], begin_norm_axis=2)
        return atom_features, p_weighted_list, p_attn_list

    def predict(self, out, mask, d_matrix):
        out_masked = out * mask
        if self.aggregation_type == 'mean':
            out_sum = fluid.layers.reduce_sum(out_masked, dim=1)
            mask_sum = fluid.layers.reduce_sum(mask, dim=1)
            out_pooling = out_sum / mask_sum
        elif self.aggregation_type == 'sum':
            out_pooling = fluid.layers.reduce_sum(out_masked, dim=1)
        elif self.aggregation_type == 'dummy_node':
            out_pooling = out_masked[:, 0]

        if self.pred_layer_num == 1:
            pred = fluid.layers.fc(out_pooling, self.output_dim, name='pred_fc')
        elif self.pred_layer_num > 1:
            pred = out_pooling
            for i in range(self.pred_layer_num - 1):
                pred = fluid.layers.fc(pred, self.hidden_size, act='relu',
                                       name='pred_fc_%d' % i)
                pred = fluid.layers.layer_norm(pred)
            pred = fluid.layers.fc(pred, self.output_dim, name='pred_fc')
        else:
            raise ValueError('Invalid value of pred_layer_num')

        return pred

    def train(self, for_attrmask_pretrain=False, task_type='cls'):
        assert task_type in ['cls', 'reg']
        graph_wrapper, pred = self.forward(
            for_attrmask_pretrain=for_attrmask_pretrain)

        if for_attrmask_pretrain:
            assert task_type == 'cls'
            masked_node_indice = fluid.layers.data(
                name="masked_node_indice", shape=[None, 1], dtype='int64')
            masked_node_label = fluid.layers.data(
                name="masked_node_label", shape=[None, 1], dtype='int64')
            masked_node_repr = fluid.layers.gather(pred, masked_node_indice)
            logits = fluid.layers.fc(
                masked_node_repr,
                size=len(CompoundConstants.atom_num_list),
                name="masked_node_logits")

            loss, pred = fluid.layers.softmax_with_cross_entropy(
                logits, masked_node_label, return_softmax=True)
        else:
            if task_type == 'cls':
                # NOTE: use one-hot label for classification, so that
                # we can compute ROC-AUC easily.
                graph_label = fluid.layers.data(
                    name='label', shape=[None, self.output_dim],
                    dtype='float32')
                loss, pred = fluid.layers.softmax_with_cross_entropy(
                    pred, graph_label, return_softmax=True, soft_label=True)
            elif task_type == 'reg':
                graph_label = fluid.layers.data(
                    name='label', shape=[None, 1], dtype='float32')
                loss = fluid.layers.square_error_cost(
                    input=pred, label=graph_label)

        self.loss = fluid.layers.reduce_mean(loss)
        self.graph_wrapper = graph_wrapper
        self.pred = pred

    def inference(self):
        self.graph_wrapper, self.pred = self.forward()
