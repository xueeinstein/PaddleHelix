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
Update the protein sequences using its PDB

NOTE: due to the limitations of experiments, it is hard to get the full
PDB structures for given protein amino acid sequences.
This script get the new protein sequences (usually a segment) from PDB files.
"""

import os
import json
import argparse
from collections import OrderedDict

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one


def main():
    """Entry for data preprocessing.

    When the input `--protein_json` is 'a/b/protein.txt', this script would
    generate 'a/b/protein_new.txt'
    """
    proteins = json.load(
        open(args.protein_json), object_pairs_hook=OrderedDict)

    pdb_parser = PDBParser()
    for pdb in os.listdir(args.pdb_dir):
        if not pdb.endswith('.pdb'):
            continue

        pdb_file = os.path.join(args.pdb_dir, pdb)
        pdb = os.path.splitext(pdb)[0]
        structure = pdb_parser.get_structure(pdb, pdb_file)
        assert len(structure) == 1
        assert pdb in proteins

        # if pdb != 'P35916':
        #     continue

        model = structure[0]
        chains = [c for c in model.get_chains()]
        aa_seqs = []

        for chain in chains:
            seq = []
            for residue in chain:
                try:
                    aa = three_to_one(residue.resname)
                    seq.append(aa)
                except:
                    pass

            aa_seqs.append(''.join(seq))

        proteins[pdb] = ''.join(aa_seqs)

    json_file, json_ext = os.path.splitext(args.protein_json)
    json_file = '{}_new{}'.format(json_file, json_ext)

    with open(json_file, 'w') as f:
        json.dump(proteins, f)
    print('Saved {}'.format(json_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', type=str, default=None, required=True)
    parser.add_argument('--protein_json', type=str, default=None, required=True)
    args = parser.parse_args()
    main()
