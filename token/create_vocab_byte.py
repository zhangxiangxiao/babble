"""Create digram vocabulary of given vocabulary size."""

from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input', 'data/tinyshakespeare/train_vocab_byte_raw.txt',
    'Input file.')
flags.DEFINE_string(
    'output', 'data/tinyshakespeare/train_vocab_byte.txt',
    'Output vocabulary file.')
flags.DEFINE_integer('size', 65536, 'Maximum vocablary size.')


def main(argv):
    # Initialize vocabulary.
    keys = []
    vocab = {}
    for i in range(256):
        key = i.to_bytes(1, 'little')
        keys.append(key)
        vocab[key] = 0
    logging.info('Load input from %s.', FLAGS.input)
    # Build the vocabulary.
    with open(FLAGS.input, 'r') as input_fd:
        while len(keys) < FLAGS.size:
            line = next(input_fd, None)
            if line == None:
                break
            key, value = line.split()
            key = bytes.fromhex(key)
            value = int(value)
            logging.info('New key %s with value %d.', key, value)
            if key not in vocab:
                keys.append(key)
            vocab[key] = value
    # Write vocabulary to file
    logging.info('Write vocabulary of size %d to %s.', len(vocab), FLAGS.output)
    with open(FLAGS.output, 'w') as output_fd:
        for key in keys:
            output_fd.write('{} {}\n'.format(key.hex(), vocab[key]))


if __name__ == '__main__':
    app.run(main)
