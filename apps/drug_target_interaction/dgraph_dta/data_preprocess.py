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
Convert Kiba and Davis datasets into npz file which can be trained directly.

Note that the dataset split is inherited from GraphDTA and DeepDTA,
the MSA and contact map data are processed by
HH-Suite (https://github.com/soedinglab/hh-suite) and
PconsC4 (https://github.com/ElofssonLab/PconsC4) respectively.
"""

import os
import json
import pickle
import random
import argparse
import numpy as np
from rdkit import Chem
from collections import OrderedDict

from pahelix.utils.protein_tools import ProteinConstants
from pahelix.utils.compound_tools import smiles_to_graph_data
from pahelix.utils.data_utils import save_data_list_to_npz


def has_cmap_and_msa(protein_key, contact_map_dir, msa_dir):
    cmap_file = os.path.join(contact_map_dir, protein_key + '.npy')
    aln_file = os.path.join(msa_dir, protein_key + '.aln')
    if os.path.exists(cmap_file) and os.path.exists(aln_file):
        return True
    else:
        return False


def one_hot_encoding(x, allowable_set):
    assert x in allowable_set
    return list(map(lambda s: x == s, allowable_set))


def amino_acid_to_feat(amino_acid):
    def _normalize(dict_):
        max_v = float(dict_[max(dict_, key=dict_.get)])
        min_v = float(dict_[min(dict_, key=dict_.get)])
        interval = max_v - min_v
        for k in dict_.keys():
            dict_[k] = (dict_[k] - min_v) / interval
        dict_['X'] = (max_v + min_v) / 2.0
        return dict_

    feat_1 = list(map(lambda set_: int(amino_acid in set_), [
        ProteinConstants.aliphatic_amino_acids,
        ProteinConstants.aromatic_amino_acids,
        ProteinConstants.polar_neutral_amino_acids,
        ProteinConstants.acidic_charged_amino_acids,
        ProteinConstants.basic_charged_amino_acids
    ]))

    feat_2 = list(map(lambda dict_: _normalize(dict_)[amino_acid], [
        ProteinConstants.amino_acids_MW,
        ProteinConstants.amino_acids_pKa,
        ProteinConstants.amino_acids_pKb,
        ProteinConstants.amino_acids_pKx,
        ProteinConstants.amino_acids_pl,
        ProteinConstants.amino_acids_hydrophobic_ph2,
        ProteinConstants.amino_acids_hydrophobic_ph7
    ]))

    return np.array(feat_1 + feat_2)


def protein_seq_to_feat(protein_key, protein_seq, msa_dir, pseudo_count=0.8,
                        without_msa=False):
    """Create protein sequence embedding using MSA and amino acid type."""
    # Compute other features from amino acid types and properties
    types = np.zeros((len(protein_seq), len(ProteinConstants.amino_acids)))
    properties = np.zeros((len(protein_seq), ProteinConstants.amino_acids_properties_dim))
    for i in range(len(protein_seq)):
        types[i, :] = one_hot_encoding(
            protein_seq[i], ProteinConstants.amino_acids)
        properties[i, :] = amino_acid_to_feat(protein_seq[i])

    if without_msa:
        joint_feat = np.concatenate([types, properties], axis=1)
    else:
        # Compute Position-Specific Scoring Matrix (PSSM)
        pfm_mat = np.zeros((len(ProteinConstants.amino_acids), len(protein_seq)))
        aln_file = os.path.join(msa_dir, protein_key + '.aln')
        with open(aln_file, 'r') as f:
            line_count = len(f.readlines())
            for line in f.readlines():
                if len(line) != len(pro_seq):
                    print('error', len(line), len(pro_seq))
                    continue
                count = 0
                for res in line:
                    if res not in pro_res_table:
                        count += 1
                        continue
                    pfm_mat[pro_res_table.index(res), count] += 1
                    count += 1

        pssm = (pfm_mat + pseudo_count / 4) / (float(line_count) + pseudo_count)
        joint_feat = np.concatenate(
            [np.transpose(pssm, (1, 0)), types, properties], axis=1)

    return joint_feat


def protein_to_graph(protein_key, protein_seq, contact_map_dir, msa_dir,
                     th=0.5, without_msa=False):
    """Convert target protein to graph using contact map and MSA."""
    contact_map = np.load(os.path.join(contact_map_dir, protein_key + '.npy'))
    contact_map += np.matrix(np.eye(contact_map.shape[0]))  # add self-loop

    edges = []
    rows, cols = np.where(contact_map >= th)
    for i, j in zip(rows, cols):
        edges.append([i, j])

    # Node feature
    seq_feat = protein_seq_to_feat(protein_key, protein_seq, msa_dir,
                                   without_msa=without_msa)

    data = {
        'protein_edges': np.array(edges),
        'protein_seq_feat': seq_feat
    }

    return data


def main():
    """Entry for data preprocessing."""
    # for dataset in ['davis', 'kiba']:
    for dataset in ['kiba']:
        data_dir = os.path.join(args.dataset_root, dataset)
        contact_map_dir = os.path.join(data_dir, args.cmap_dir)
        msa_dir = os.path.join(data_dir, 'aln')

        all_found = sum([os.path.exists(d) for d in
                         [data_dir, contact_map_dir, msa_dir]]) == 3
        if not all_found:
            print('Cannot find all required data '
                  'folders for {}'.format(dataset))
            continue

        train_fold = json.load(
            open(os.path.join(data_dir, 'folds', 'train_fold_setting1.txt')))
        train_fold = [ee for e in train_fold for ee in e]  # flatten
        test_fold = json.load(
            open(os.path.join(data_dir, 'folds', 'test_fold_setting1.txt')))
        ligands = json.load(
            open(os.path.join(data_dir, 'ligands_can.txt')),
            object_pairs_hook=OrderedDict)
        proteins = json.load(
            open(os.path.join(data_dir, args.protein_json)),
            object_pairs_hook=OrderedDict)
        # Use encoding 'latin1' to load py2 pkl from py3
        # pylint: disable=E1123
        affinity = pickle.load(
            open(os.path.join(data_dir, 'Y'), 'rb'), encoding='latin1')

        smiles_lst, protein_lst, protein_keys = [], [], []
        for k in ligands.keys():
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[k]),
                                      isomericSmiles=True)
            smiles_lst.append(smiles)

        for k in proteins.keys():
            protein_lst.append(proteins[k])
            protein_keys.append(k)

        if dataset == 'davis':
            # Kd data
            affinity = [-np.log10(y / 1e9) for y in affinity]

        affinity = np.asarray(affinity)

        # pylint: disable=E1123
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
        for split in ['train', 'test']:
            print('processing {} set of {}'.format(split, dataset))

            split_dir = os.path.join(data_dir, 'processed', split)
            # pylint: disable=E1123
            os.makedirs(split_dir, exist_ok=True)

            fold = train_fold if split == 'train' else test_fold
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[fold], cols[fold]

            data_lst = [[] for _ in range(args.npz_files)]
            for idx in range(len(rows)):
                if not has_cmap_and_msa(
                        protein_keys[cols[idx]], contact_map_dir, msa_dir):
                    continue

                mol_graph = smiles_to_graph_data(smiles_lst[rows[idx]])
                data = {k: v for k, v in mol_graph.items()}

                prot_graph = protein_to_graph(protein_keys[cols[idx]],
                                              protein_lst[cols[idx]],
                                              contact_map_dir, msa_dir,
                                              without_msa=args.without_msa)
                for k, v in prot_graph.items():
                    data[k] = v

                af = affinity[rows[idx], cols[idx]]
                if dataset == 'davis':
                    data['Log10_Kd'] = np.array([af])
                elif dataset == 'kiba':
                    data['KIBA'] = np.array([af])

                data_lst[idx % args.npz_files].append(data)

                if idx % 100 == 0:
                    print('processed #{}/{} {} set of {}'.format(
                        idx, len(rows), split, dataset))

            print('processed #{}/{} {} set of {}'.format(
                idx, len(rows), split, dataset))

            random.shuffle(data_lst)
            for j, sub_data_lst in enumerate(data_lst):
                random.shuffle(sub_data_lst)
                npz = os.path.join(
                    split_dir, '{}_{}_{}.npz'.format(dataset, split, j))
                save_data_list_to_npz(sub_data_lst, npz)

        print('==============================')
        print('dataset:', dataset)
        print('train_fold:', len(train_fold))
        print('test_fold:', len(test_fold))
        print('unique drugs:', len(set(smiles_lst)))
        print('unique proteins:', len(set(protein_lst)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default=None, required=True)
    parser.add_argument('--protein_json', type=str, default='proteins.txt')
    parser.add_argument('--npz_files', type=int, default=1)  # set it > 1 for multi trainers
    # it's optional to use ground truth contact map
    parser.add_argument('--cmap_dir', type=str, default='pconsc4')
    parser.add_argument('--without_msa', default=False, action='store_true')
    args = parser.parse_args()
    main()
