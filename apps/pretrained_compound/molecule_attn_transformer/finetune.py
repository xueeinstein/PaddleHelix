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

import paddle
from paddle import fluid

from pahelix.datasets import InMemoryDataset
from pahelix.utils.splitters import RandomSplitter
from pahelix.utils.data_utils import load_npz_to_data_list
from pahelix.utils.paddle_utils import load_partial_params

from model import GraphTransformer
from data_gen import MoleculeCollateFunc
from utils import default_exe_params, default_optimizer, setup_optimizer, \
    split_distributed_dataset

logging.basicConfig(
        format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)

Regression_Tasks = ['esol', 'freesolv']


def is_reg_task(data_npz):
    task = os.path.splitext(os.path.basename(data_npz))[0]
    return task.lower() in Regression_Tasks


def train(args, exe, train_program, model, train_dataset):
    """Model training for one epoch and return the average loss."""
    collate_fn = MoleculeCollateFunc(
        model.graph_wrapper,
        with_graph_label=True,
        with_attr_mask=False,
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
        with_graph_label=True,
        with_attr_mask=False,
        label_dim=model.output_dim)

    val_loss, test_loss = [], []
    for feed_dict in val_dataset.iter_batch(
            args.batch_size, num_workers=args.num_workers,
            shuffle=True, collate_fn=collate_fn):
        loss, = exe.run(
            test_program, feed=feed_dict, fetch_list=[model.loss],
            return_numpy=False)
        val_loss.append(np.array(loss))

    val_loss = np.mean(val_loss)
    if val_loss < best_val_loss:
        for feed_dict in test_dataset.iter_batch(
                args.batch_size, num_workers=args.num_workers,
                shuffle=True, collate_fn=collate_fn):
            loss, = exe.run(
                test_program, feed=feed_dict, fetch_list=[model.loss],
                return_numpy=False)
            test_loss.append(np.array(loss))

        test_loss = np.mean(test_loss)
        return val_loss, test_loss
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

    train_program, train_startup = fluid.Program(), fluid.Program()
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            task_type = 'reg' if is_reg_task(args.data_npz) else 'cls'
            model = GraphTransformer(model_config, name='compound')
            model.train(for_attrmask_pretrain=False, task_type=task_type)

            test_program = train_program.clone(for_test=True)
            optimizer = default_optimizer(args.lr, warmup_steps=args.warmup_steps)
            setup_optimizer(optimizer, args.use_cuda, args.is_distributed,
in_startup)
    if args.init_model is not None and args.init_model != "":
        load_partial_params(exe, args.init_model, train_program)

    config = os.path.basename(args.model_config)
    best_val_loss, best_test_loss, best_ep = np.inf, np.inf, 0
    best_model = os.path.join(args.model_dir, 'best_model')
    for epoch_id in range(1, args.max_epoch + 1):
        logging.info('========== Epoch {} =========='.format(epoch_id))
        train_loss = train(args, exe, train_program, model, train_dataset)
        logging.info('#{} Epoch: {}, Train loss: {}'.format(
            config, epoch_id, train_loss))

        val_loss, test_loss = evaluate(
            args, exe, test_program, model, val_dataset, test_dataset,
            best_val_loss)

        if trainer_id == 0 and val_loss < best_val_loss:
            best_ep = epoch_id
            best_val_loss = val_loss
            best_test_loss = test_loss

            if os.path.exists(best_model):
                shutil.rmtree(best_model)
            fluid.io.save_params(exe, best_model, train_program)

        if trainer_id == 0:
            logging.info('Best val loss: {}, test loss: {}'.format(
                best_val_loss, best_test_loss))
            if best_ep < epoch_id:
                logging.info('No improvement since epoch {}'.format(best_ep))


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
    parser.add_argument("--data_npz", type=str)
    parser.add_argument("--data_split", type=str)

    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()
    main(args)
