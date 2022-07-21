"""Tokenize using HuggingFace tokenizers."""

import h5py
from absl import app
from absl import flags
from absl import logging
from tokenizers import Tokenizer


FLAGS = flags.FLAGS
flags.DEFINE_string('tokenizer', 'data/tinyshakespeare/bpe.json',
                    'Tokenizer file.')
flags.DEFINE_string('input', 'data/tinyshakespeare/train.h5',
                    'Input HDF5 file.')
flags.DEFINE_string('output', 'data/tinyshakespeare/train_bpe.h5',
                    'Output HDF5 file.')


def initialize_tokenizer(filename):
    return Tokenizer.from_file(filename)


def tokenize(index, content, tokenizer, filename):
    pass


def main(unused_argv):
    pass


if __name__ == '__main__':
    app.run(main)

