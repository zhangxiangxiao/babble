# Train a tokenizer using HuggingFace.

import h5py
import numpy as np
from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'data/tinyshakespeare/train.h5',
                    'Input HDF5 file.')
flags.DEFINE_enum('model', 'bpe', ['bpe'], 'The tokenizer model to use')
flags.DEFINE_string('output', 'data/tinyshakespeare/bpe.json',
                    'Output trained tokenizer model.')


def main(unused_argv):
    pass


if __name__ == '__main__':
    app.run(main)
