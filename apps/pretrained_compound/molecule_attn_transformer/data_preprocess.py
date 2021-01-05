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
Convert BBBP, ESOL, FreeSolv, Estrogen-alpha, Estrogen-beta, Mesta-high,
and Mesta-low datasets to npz files.
"""

import os
import glob
import argparse
import threading
import numpy as np
import pandas as pd
from queue import Queue
from rdkit.Chem import AllChem

from pahelix.datasets import InMemoryDataset
from pahelix.utils.splitters import RandomSplitter
from pahelix.utils.data_utils import save_data_list_to_npz
from pahelix.utils.compound_tools import smiles_to_graph_data

N_split = 8


def process_mol_graph(smiles):
    # smiles = AllChem.MolToSmiles(
    #     AllChem.MolFromSmiles(smiles), isomericSmiles=False)
    mol_graph = smiles_to_graph_data(
        smiles, get_adj_dist=True,
        add_dummy_node=not args.no_dummy_node)
    data = {k: v for k, v in mol_graph.items()}
    d = data['adj_matrix'].shape[0]
    data['adj_matrix'] = np.reshape(data['adj_matrix'], (d*d,))
    data['dist_matrix'] = np.reshape(data['dist_matrix'], (d*d,))
    data['matrix_d'] = np.array([d])
    return data


def process_zinc_dataset(csv, num_workers):
    """Process raw ZINC dataset for self-supervised pretraining."""
    def _worker_fn(queue, idx_list, smiles_list):
        for idx, smiles in zip(idx_list, smiles_list):
            data = process_mol_graph(smiles)
            queue.put((idx, data))

    df = pd.read_csv(csv)
    smiles_list = list(df.smiles)
    idx_list = [i for i in range(len(df.smiles))]

    queue = Queue(100)
    for wid in range(num_workers):
        sub_idx_list = [i for i in idx_list if i % num_workers == wid]
        sub_smiles_list = [smiles_list[i] for i in sub_idx_list]
        worker = threading.Thread(
            target=_worker_fn,
            args=(queue, sub_idx_list, sub_smiles_list))
        worker.setDaemon = True
        worker.start()

    recv, pairs = 0, []
    csv = os.path.basename(csv)
    while recv < len(df.smiles):
        idx, data = queue.get()
        # assert data['matrix_d'][0] == data['atom_type'].size + 1
        if data['matrix_d'][0] != data['atom_type'].size + 1:
            import ipdb; ipdb.set_trace()
        pairs.append((idx, data))
        recv += 1

        if recv % 1000 == 0:
            print('{} processed {}/{}'.format(csv, recv, len(df.smiles)))
    print('{} processed {}/{}'.format(csv, recv, len(df.smiles)))

    pairs = sorted(pairs, key=lambda i: i[0])
    return [i[1] for i in pairs]


def process_downtask_datasets(csv, num_workers):
    """Process downtasks datasets with labels."""
    def _worker_fn(queue, idx_list, smiles_list, y_list):
        for idx, smiles, y in zip(idx_list, smiles_list, y_list):
            data = process_mol_graph(smiles)
            data['label'] = np.array([y])
            queue.put((idx, data))

    df = pd.read_csv(csv)
    smiles_list, y_list = list(df.smiles), list(df.y)
    idx_list = [i for i in range(len(df.y))]

    queue = Queue(100)
    for wid in range(num_workers):
        sub_idx_list = [i for i in idx_list if i % num_workers == wid]
        sub_smiles_list = [smiles_list[i] for i in sub_idx_list]
        sub_y_list = [y_list[i] for i in sub_idx_list]
        worker = threading.Thread(
            target=_worker_fn,
            args=(queue, sub_idx_list, sub_smiles_list, sub_y_list))
        worker.setDaemon = True
        worker.start()

    recv, pairs = 0, []
    csv = os.path.basename(csv)
    while recv < len(df.y):
        pairs.append(queue.get())
        recv += 1

        if recv % 1000 == 0:
            print('{} processed {}/{}'.format(csv, recv, len(df.y)))
    print('{} processed {}/{}'.format(csv, recv, len(df.y)))

    pairs = sorted(pairs, key=lambda i: i[0])
    return [i[1] for i in pairs]


def main():
    """Entry for data preprocessing."""
    for ds in os.listdir(args.dataset_root):
        print('Processing {}'.format(ds))
        csv_files = glob.glob(os.path.join(args.dataset_root, ds, '*.csv'))
        if len(csv_files) == 0:
            print('Cannot found csv file for {}'.format(ds))
        elif len(csv_files) > 1:
            print('There are more than one csv files for {}'.format(ds))
        else:
            if args.for_zinc:
                ds = 'zinc'
                data_list = process_zinc_dataset(csv_files[0], args.num_workers)
                # from pahelix.utils.data_utils import load_npz_to_data_list
                # data_list = load_npz_to_data_list(
                #     '/mnt/xueyang/Datasets/PaddleHelix/mat/zinc.npz')
                splitter = RandomSplitter()
                train_dataset, val_dataset, test_dataset = splitter.split(
                    InMemoryDataset(data_list),
                    frac_train=0.8, frac_valid=0.1, frac_test=0.1)

                n = len(train_dataset)
                for s in range(N_split):
                    indices = [i for i in range(n) if i % N_split == s]
                    npz = os.path.join(
                        args.output_dir, '{}_train_{}.npz'.format(ds, s))
                    save_data_list_to_npz(
                        train_dataset[indices].data_list, npz)
                    print('Saved {}'.format(npz))

                npz = os.path.join(args.output_dir, '{}_val.npz'.format(ds))
                save_data_list_to_npz(val_dataset.data_list, npz)
                print('Saved {}'.format(npz))

                npz = os.path.join(args.output_dir, '{}_test.npz'.format(ds))
                save_data_list_to_npz(test_dataset.data_list, npz)
                print('Saved {}'.format(npz))

            else:
                data_list = process_downtask_datasets(csv_files[0], args.num_workers)

                npz = os.path.join(args.output_dir, '{}.npz'.format(ds))
                save_data_list_to_npz(data_list, npz)
                print('Saved {}'.format(npz))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default=None, required=True)
    parser.add_argument('--no_dummy_node', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--for_zinc', default=False, action='store_true')
    args = parser.parse_args()
    main()
