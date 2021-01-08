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
Tools for protein features.
"""

from collections import OrderedDict
from enum import Enum


class ProteinConstants(object):
    """
    Constants of amino acids of protein.

    Reference: https://www.sigmaaldrich.com/life-science/metabolomics/learning-center/amino-acid-reference-chart.html
    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                   'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

    aliphatic_amino_acids = ['A', 'I', 'L', 'M', 'V']
    aromatic_amino_acids = ['F', 'W', 'Y']
    polar_neutral_amino_acids = ['C', 'N', 'Q', 'S', 'T']
    acidic_charged_amino_acids = ['D', 'E']
    basic_charged_amino_acids = ['H', 'K', 'R']

    # molecular weights
    amino_acids_MW = {
        'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18,
        'G': 57.05, 'H': 137.14, 'I': 113.16, 'K': 128.18, 'L': 113.16,
        'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13, 'R': 156.19,
        'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18
    }

    # negative of the logarithm of the dissociation constant
    # for the -COOH group
    amino_acids_pKa = {
        'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83,
        'G': 2.34, 'H': 1.82, 'I': 2.36, 'K': 2.18, 'L': 2.36,
        'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17,
        'S': 2.21, 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32
    }

    # negative of the logarithm of the dissociation constant
    # for the -NH3 group
    amino_acids_pKb = {
        'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13,
        'G': 9.60, 'H': 9.17, 'I': 9.60, 'K': 8.95, 'L': 9.60,
        'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13, 'R': 9.04,
        'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62
    }

    # negative of the logarithm of the dissociation constant
    # for any other group in the molecule
    amino_acids_pKx = {
        'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00,
        'G': 0.00, 'H': 6.00, 'I': 0.00, 'K': 10.53, 'L': 0.00,
        'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00, 'R': 12.48,
        'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00
    }

    # pH at the isoelectric point
    amino_acids_pl = {
        'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48,
        'G': 5.97, 'H': 7.59, 'I': 6.02, 'K': 9.74, 'L': 5.98,
        'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76,
        'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96
    }

    amino_acids_hydrophobic_ph2 = {
        'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92,
        'G': 0, 'H': -42, 'I': 100, 'K': -37, 'L': 100,
        'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26,
        'S': -7, 'T': 13, 'V': 79, 'W': 84, 'Y': 49
    }

    amino_acids_hydrophobic_ph7 = {
        'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100,
        'G': 0, 'H': 8, 'I': 99, 'K': -23, 'L': 97,
        'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14,
        'S': -5, 'T': 13, 'V': 76, 'W': 97, 'Y': 63
    }

    amino_acids_properties_dim = 12  # TODO: make dynamic binding


class ProteinTokenizer(object):
    """
    Protein Tokenizer.
    """
    padding_token = '<pad>'
    mask_token = '<mask>'
    start_token = class_token = '<cls>'
    end_token = seperate_token = '<sep>'
    unknown_token = '<unk>'

    padding_token_id = 0
    mask_token_id = 1
    start_token_id = class_token_id = 2
    end_token_id = seperate_token_id = 3
    unknown_token_id = 4

    special_token_ids = [padding_token_id, mask_token_id, start_token_id, end_token_id, unknown_token_id]

    vocab = OrderedDict([
        (padding_token, 0),
        (mask_token, 1),
        (class_token, 2),
        (seperate_token, 3),
        (unknown_token, 4),
        ('A', 5),
        ('B', 6),
        ('C', 7),
        ('D', 8),
        ('E', 9),
        ('F', 10),
        ('G', 11),
        ('H', 12),
        ('I', 13),
        ('K', 14),
        ('L', 15),
        ('M', 16),
        ('N', 17),
        ('O', 18),
        ('P', 19),
        ('Q', 20),
        ('R', 21),
        ('S', 22),
        ('T', 23),
        ('U', 24),
        ('V', 25),
        ('W', 26),
        ('X', 27),
        ('Y', 28),
        ('Z', 29)])

    def tokenize(self, sequence):
        """
        Split the sequence into token list.

        Args:
            sequence: The sequence to be tokenized.

        Returns:
            tokens: The token lists.
        """
        return [x for x in sequence]

    def convert_token_to_id(self, token):
        """ 
        Converts a token to an id.

        Args:
            token: Token.

        Returns:
            id: The id of the input token.
        """
        if token not in self.vocab:
            return ProteinTokenizer.unknown_token_id
        else:
            return ProteinTokenizer.vocab[token]

    def convert_tokens_to_ids(self, tokens):
        """
        Convert multiple tokens to ids.
        
        Args:
            tokens: The list of tokens.

        Returns:
            ids: The id list of the input tokens.
        """
        return [self.convert_token_to_id(token) for token in tokens]

    def gen_token_ids(self, sequence):
        """
        Generate the list of token ids according the input sequence.

        Args:
            sequence: Sequence to be tokenized.

        Retuens:
            token_ids: The list of token ids.
        """
        tokens = []
        tokens.append(ProteinTokenizer.start_token)
        tokens.extend(self.tokenize(sequence))
        tokens.append(ProteinTokenizer.end_token)
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids


