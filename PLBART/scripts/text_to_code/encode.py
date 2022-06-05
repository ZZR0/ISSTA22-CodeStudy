#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from tqdm import tqdm

from multiprocessing import Pool
import sentencepiece as spm


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global sp
        sp = spm.SentencePieceProcessor(model_file=self.args.model_file)

    def _encode(self, line):
        global sp
        return sp.encode(line, out_type=str)

    def _decode(self, tokens):
        global sp
        return sp.decode(tokens)

    def encode(self, example):
        assert isinstance(example, dict)
        assert 'src' in example and 'tgt' in example
        if example['src'] is None:
            return None
        if example['tgt'] is None:
            return None
        if len(example['src']) == 0 and not self.args.keep_empty:
            return None
        if len(example['tgt']) == 0 and not self.args.keep_empty:
            return None
        src_tokens = self._encode(example['src'])[:self.args.max_len]
        tgt_tokens = self._encode(example['tgt'])[:self.args.max_len]
        return {'src': " ".join(src_tokens), 'tgt': " ".join(tgt_tokens)}


def load_data(input_file, src_field, tgt_field):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line.strip())
            assert src_field in ex and tgt_field in ex
            src = ex[src_field]
            if isinstance(src, list):
                src = " ".join(src)
            tgt = ex[tgt_field]
            if isinstance(tgt, list):
                tgt = " ".join(tgt)
            data.append({'src': src, 'tgt': tgt})

    return data


def process(args):
    dataset = load_data(args.input_file, args.src_field, args.tgt_field)

    encoder = MultiprocessingEncoder(args)
    pool = Pool(args.workers, initializer=encoder.initializer)

    processed_dataset = []
    with tqdm(total=len(dataset), desc='Processing') as pbar:
        for i, ex in enumerate(pool.imap(encoder.encode, dataset, 100)):
            pbar.update()
            processed_dataset.append(ex)

    out_src = os.path.join(args.output_dir, '{}.spm.{}'.format(args.pref, args.src_lang))
    out_tgt = os.path.join(args.output_dir, '{}.spm.{}'.format(args.pref, args.tgt_lang))
    with open(out_src, 'w', encoding='utf-8') as src_writer, \
            open(out_tgt, 'w', encoding='utf-8') as tgt_writer:
        for ex in processed_dataset:
            if ex is not None:
                src_writer.write(ex['src'] + '\n')
                tgt_writer.write(ex['tgt'] + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-file",
        help='path to *.model file',
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=['-'],
        help="input files (.jsonl) to filter/encode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=['_'],
        help="path of the output directory",
    )
    parser.add_argument(
        "--src_field",
        type=str,
        default=['_'],
        help="field name to be considered as the source",
    )
    parser.add_argument(
        "--tgt_field",
        type=str,
        default=['_'],
        help="field name to be considered as the source",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default=['_'],
        help="name of the source language",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default=['_'],
        help="name of the target language",
    )
    parser.add_argument(
        "--pref",
        type=str,
        default=['_'],
        help="file prefix",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--max_len", type=int, default=510)
    parser.add_argument("--workers", type=int, default=60)
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
