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
utils
"""
import os
import logging
import numpy as np
from sklearn.metrics import roc_auc_score

from paddle import fluid
from paddle.fluid.incubate.fleet.base import role_maker
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy


def default_exe_params(is_distributed, use_cuda, thread_num):
    """
    Set the default execute parameters.
    """
    gpu_id = 0
    trainer_num = 1
    trainer_id = 0
    dist_strategy = None
    places = None
    if is_distributed:
        if use_cuda:
            role = role_maker.PaddleCloudRoleMaker(is_collective=True)
            fleet.init(role)

            gpu_id = int(os.getenv("FLAGS_selected_gpus"))
            trainer_num = fleet.worker_num()
            trainer_id = fleet.worker_index()

            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.use_experimental_executor = True
            exec_strategy.num_threads = 4
            exec_strategy.num_iteration_per_drop_scope = 1

            dist_strategy = DistributedStrategy()
            dist_strategy.exec_strategy = exec_strategy
            dist_strategy.nccl_comm_num = 2
            dist_strategy.fuse_all_reduce_ops = True

            dist_strategy.forward_recompute = True

            dist_strategy.use_amp = True
            dist_strategy.amp_loss_scaling = 12800.0

            places = fluid.cuda_places()
        else:
            print('Only gpu is supported for distributed mode at present.')
            exit(-1)
    else:
        if use_cuda:
            places = fluid.cuda_places()
        else:
            places = fluid.cpu_places(thread_num)
            os.environ['CPU_NUM'] = str(thread_num)

    if use_cuda:
        exe = fluid.Executor(fluid.CUDAPlace(gpu_id))
    else:
        exe = fluid.Executor(fluid.CPUPlace())

    return {
            'exe': exe,
            'trainer_num': trainer_num,
            'trainer_id': trainer_id,
            'gpu_id': gpu_id,
            'dist_strategy': dist_strategy,
            'places': places}


def default_optimizer(lr, warmup_steps=0):
    """
    Get default Adam optimizer.
    """
    if warmup_steps > 0:
        scheduled_lr = fluid.layers.learning_rate_scheduler.noam_decay(
            1 / (warmup_steps * (lr ** 2)), warmup_steps)
    else:
        scheduled_lr = fluid.layers.create_global_var(
            name=fluid.unique_name.generate("learning_rate"),
            shape=[1],
            value=lr,
            dtype="float32",
            persistable=True)

    optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
    return optimizer


def setup_optimizer(optimizer, use_cuda, is_distributed, dist_strategy):
    """
    Setup the optimizer
    """
    if use_cuda:
        if is_distributed:
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=dist_strategy)


def split_distributed_dataset(dataset, trainer_id, trainer_num):
    """
    Split dataset for distributed trainers
    """
    assert trainer_num > 1 and trainer_id < trainer_num
    N = len(dataset)
    indices = [i for i in range(N) if i % trainer_num == trainer_id]
    return dataset[indices]


def calc_rocauc_score(labels, preds):
    """compute ROC-AUC and averaged across tasks
    """
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)

    rocauc_list = []
    for i in range(labels.shape[1]):
        c_label, c_pred = labels[:, i], preds[:, i]
        #AUC is only defined when there is at least one positive data.
        if len(np.unique(c_label)) == 2:
            rocauc_list.append(roc_auc_score(c_label, c_pred))

    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

    return sum(rocauc_list)/len(rocauc_list)
