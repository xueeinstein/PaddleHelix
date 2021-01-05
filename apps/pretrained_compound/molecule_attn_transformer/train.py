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
Script to train and eval representations from MAT
"""

import os
import sys
import glob
import json
import shutil
import logging
import argparse
import numpy as np
from collections import namedtuple
from sklearn.metrics import accuracy_score

import paddle
from paddle import fluid

from pahelix.datasets import InMemoryDataset
from pahelix.utils.splitters import RandomSplitter
from pahelix.utils.data_utils import load_npz_to_data_list
from pahelix.utils.paddle_utils import load_partial_params

from model import GraphTransformer
from data_gen import MoleculeCollateFunc
from utils import default_exe_params, default_optimizer, setup_optimizer, \
    split_distributed_dataset, calc_rocauc_score

logging.basicConfig(
        format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)

Regression_Tasks = ['esol', 'freesolv']


def is_reg_task(data_npz):
    task = os.path.splitext(os.path.basename(data_npz))[0]
    return task.lower() in Regression_Tasks


def get_eval_metric(task_type, output_dim):
    if task_type == 'reg':
        return 'RMSE'
    elif task_type == 'cls':
        if output_dim == 2:
            return 'ROC-AUC'
        else:
            return 'ACC'


def train(args, exe, train_program, model, train_dataset):
    """Model training for one epoch and return the average loss."""
    collate_fn = MoleculeCollateFunc(
        model.graph_wrapper,
        with_graph_label=not args.pretrain,
        with_attr_mask=args.pretrain,
        label_dim=model.output_dim)

    list_loss = []
    for feed_dict in train_dataset.iter_batch(
            args.batch_size, num_workers=args.num_workers,
            shuffle=True, collate_fn=collate_fn):
        # print('===========')
        # for k, v in feed_dict.items():
        #     if k == 'd_matrix':
        #         print(k, v)
        #     elif k == 'graph/graph_lod':
        #         v_ = [v[i] - v[i-1] for i in range(1, len(v))]
        #         print(k, max(v_))
        #     else:
        #         print(k, v.shape)
        train_loss, = exe.run(
            train_program, feed=feed_dict, fetch_list=[model.loss],
            return_numpy=False)
        list_loss.append(np.array(train_loss))

    return np.mean(list_loss)


def evaluate(args, exe, test_program, model, val_dataset, test_dataset, best_val_loss):
    """Evaluate the model on the test dataset and return val & eval loss."""
    collate_fn = MoleculeCollateFunc(
        model.graph_wrapper,
        with_graph_label=not args.pretrain,
        with_attr_mask=args.pretrain,
        label_dim=model.output_dim)

    val_loss = []
    for feed_dict in val_dataset.iter_batch(
            args.batch_size, num_workers=args.num_workers,
            shuffle=True, collate_fn=collate_fn):
        loss, = exe.run(
            test_program, feed=feed_dict, fetch_list=[model.loss],
            return_numpy=False)
        val_loss.append(np.array(loss))
    val_loss = np.mean(val_loss)

    label_name = 'label' if not args.pretrain else 'masked_node_label'
    task_type = 'reg' if model.output_dim == 1 else 'cls'

    if val_loss < best_val_loss:
        test_pred, test_label = [], []
        for feed_dict in test_dataset.iter_batch(
                args.batch_size, num_workers=args.num_workers,
                shuffle=True, collate_fn=collate_fn):
            pred, = exe.run(
                test_program, feed=feed_dict, fetch_list=[model.pred],
                return_numpy=False)
            test_pred.append(np.array(pred))
            test_label.append(feed_dict[label_name])

        if task_type == 'reg':
            # Report RMSE
            test_pred = np.concatenate(test_pred, 0).flatten()
            test_label = np.concatenate(test_label, 0).flatten()
            mse = ((test_label - test_pred) ** 2).mean(axis=0)
            test_metric = np.sqrt(mse)
        elif task_type == 'cls':
            # Report ROC-AUC or Acc
            test_pred = np.concatenate(test_pred, 0)
            test_label = np.concatenate(test_label, 0)
            try:
                test_metric = calc_rocauc_score(test_label, test_pred)
            except:
                test_pred = np.argmax(test_pred, axis=1)
                test_metric = accuracy_score(test_label, test_pred)

        return val_loss, test_metric
    else:
        return val_loss, None


def main(args):
    if paddle.__version__.startswith('2.'):
        # Enable static graph mode.
        paddle.enable_static()

    model_config = json.load(open(args.model_config, 'r'))

    exe_params = default_exe_params(args.is_distributed, args.use_cuda, args.thread_num)
    exe = exe_params['exe']
    trainer_num = exe_params['trainer_num']
    trainer_id = exe_params['trainer_id']
    dist_strategy = exe_params['dist_strategy']

    # data_list = load_npz_to_data_list(args.data_npz)
    # logging.info('Loaded dataset #{}'.format(len(data_list)))

    # save a small subset for debug
    # max_i, max_d = 0, 0
    # for i, data in enumerate(data_list):
    #     if data['matrix_d'][0] > max_d:
    #         max_i, max_d = i, data['matrix_d'][0]
    # from pahelix.utils.data_utils import save_data_list_to_npz
    # if max_i > 10000:
    #     save_data_list_to_npz(data_list[max_i-10000+1:max_i+1], 'small.npz')
    # else:
    #     save_data_list_to_npz(data_list[max_i:max_i+10000], 'small.npz')
    # import ipdb; ipdb.set_trace()

    if args.pretrain:
        train_npz = glob.glob(
            os.path.join(args.data_dir, '*_train_%d.npz' % trainer_id))[0]
        train_dataset = InMemoryDataset(load_npz_to_data_list(train_npz))

        val_npz = glob.glob(os.path.join(args.data_dir, '*_val.npz'))[0]
        val_dataset = InMemoryDataset(load_npz_to_data_list(val_npz))

        test_npz = glob.glob(os.path.join(args.data_dir, '*_test.npz'))[0]
        test_dataset = InMemoryDataset(load_npz_to_data_list(test_npz))
    else:
        data_list = load_npz_to_data_list(args.data_npz)
        train_split, val_split, test_split = np.load(
            args.data_split, allow_pickle=True)

        train_dataset = InMemoryDataset([data_list[i] for i in train_split])
        val_dataset = InMemoryDataset([data_list[i] for i in val_split])
        test_dataset = InMemoryDataset([data_list[i] for i in test_split])

    logging.info('Loaded train #{}, val #{}, test #{}'.format(
        len(train_dataset), len(val_dataset), len(test_dataset)))

    if args.is_distributed:
        train_dataset = split_distributed_dataset(
            train_dataset, trainer_id, trainer_num)
        val_dataset = split_distributed_dataset(
            val_dataset, trainer_id, trainer_num)
        test_dataset = split_distributed_dataset(
            test_dataset, trainer_id, trainer_num)

    task_type = 'cls'
    if not args.pretrain and is_reg_task(args.data_npz):
        task_type = 'reg'

    train_program, train_startup = fluid.Program(), fluid.Program()
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            model = GraphTransformer(model_config, name='compound')
            model.train(for_attrmask_pretrain=args.pretrain,
                        task_type=task_type)

            test_program = train_program.clone(for_test=True)
            optimizer = default_optimizer(args.lr, warmup_steps=args.warmup_steps)
            setup_optimizer(optimizer, args.use_cuda, args.is_distributed,
                            dist_strategy)

    exe.run(train_startup)
    if args.init_model is not None and args.init_model != "":
        load_partial_params(exe, args.init_model, train_program)

    config = os.path.basename(args.model_config)
    best_val_loss, best_ep = np.inf, 0
    best_test_metric = np.inf if task_type == 'reg' else 0
    best_model = os.path.join(args.model_dir, 'best_model')  # min val loss
    best_test_model = os.path.join(args.model_dir, 'best_test_model')  # best test metric
    metric_log = os.path.join(args.model_dir, 'metric.txt')
    metric = get_eval_metric(task_type, model.output_dim)
    for epoch_id in range(1, args.max_epoch + 1):
        logging.info('========== Epoch {} =========='.format(epoch_id))
        train_loss = train(args, exe, train_program, model, train_dataset)
        logging.info('#{} Epoch: {}, Train loss: {}'.format(
            config, epoch_id, train_loss))

        val_loss, test_metric = evaluate(
            args, exe, test_program, model, val_dataset, test_dataset,
            best_val_loss)

        if trainer_id == 0 and val_loss < best_val_loss:
            best_ep, best_val_loss = epoch_id, val_loss
            if os.path.exists(best_model):
                shutil.rmtree(best_model)
            fluid.io.save_params(exe, best_model, train_program)

        if trainer_id == 0:
            if (task_type == 'reg' and test_metric and best_test_metric > test_metric) or \
               (task_type == 'cls' and test_metric and best_test_metric < test_metric):
                best_test_metric = test_metric
                fluid.io.save_params(exe, best_test_model, train_program)

            logging.info('Min val loss: {}, Best test {}: {}, test {}: {}'.format(
                best_val_loss, metric, best_test_metric, metric, test_metric))
            if best_ep < epoch_id:
                logging.info('No improvement since epoch {}'.format(best_ep))

            with open(metric_log, 'w') as f:
                f.write('Val loss: {}\n'.format(best_val_loss))
                f.write('Test {}: {}\n'.format(metric, best_test_metric))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action='store_true', default=False)
    parser.add_argument("--distributed", dest='is_distributed', action='store_true')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8, help='data workers')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--thread_num", type=int, default=8, help='thread for cpu')
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--data_npz", type=str)
    parser.add_argument("--data_split", type=str)

    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--pretrain", action='store_true', default=False)
    args = parser.parse_args()
    main(args)
