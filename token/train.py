"""Train a tokenizer using HuggingFace."""

import h5py
import numpy as np
from absl import app
from absl import flags
from absl import logging
from tokenizers import Tokenizer
from tokenizers.normalizers import NFKC
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'data/tinyshakespeare/train.h5',
                    'Input HDF5 file.')
flags.DEFINE_string('output', 'data/tinyshakespeare/bpe.json',
                    'Output trained tokenizer model.')
flags.DEFINE_enum('model', 'bpe', ['bpe'], 'The tokenizer model to use.')
flags.DEFINE_integer('vocab_size', 65536, 'Vocubulary size.')
flags.DEFINE_boolean('cache', False, 'Whether to cache the input in memory.')


def initialize_tokenizer():
    if FLAGS.model == 'bpe':
        tokenizer = Tokenizer(BPE())
        tokenizer.normalizer = NFKC()
        trainer = BpeTrainer(vocab_size=FLAGS.vocab_size,
                             initial_alphabet=[chr(i) for i in range(256)],
                             show_progress=False)
        return tokenizer, trainer


def get_iterator(index, content):
    for i in range(index.shape[0]):
        content_index = index[i, 0]
        content_length = index[i, 1]
        sample_bytes = content[content_index:(content_index + content_length)]
        sample_text = sample_bytes.astype('uint8').tobytes().decode(
            'utf-8', 'replace')
        if i % 10000 == 0:
            logging.info('Processing sample %d.', i)
        yield sample_text


def main(unused_argv):
    tokenizer, trainer = initialize_tokenizer()
    logging.info('Train a %s tokenizer.', FLAGS.model)
    logging.info('Load input from %s.', FLAGS.input)
    data_fd = h5py.File(FLAGS.input, 'r')
    index = data_fd['index']
    content = data_fd['content']
    if FLAGS.cache:
        logging.info('Cache the input in memory.')
        index = index[:]
        content = content[:]
    iterator = get_iterator(index, content)
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    logging.info('Processed samples %d.', index.shape[0])
    data_fd.close()
    logging.info('Save the tokenizer to %s.', FLAGS.output)
    tokenizer.save(FLAGS.output)


if __name__ == '__main__':
    app.run(main)
