"""Create digram vocabulary of given vocabulary size."""

from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input', 'data/tinyshakespeare/train_vocab_byte_count.txt',
    'Input file.')
flags.DEFINE_string(
    'output', 'data/tinyshakespeare/train_vocab_byte.txt',
    'Output vocabulary file.')
flags.DEFINE_integer('size', 65536, 'Vocablary size.')


def main(argv):
    # Initialize vocabulary.
    left_count = {}
    right_count = {}
    vocab = {}
    for i in range(256):
        key = i.to_bytes(1, 'little')
        vocab[key] = 0
    logging.info('Load input from %s.', FLAGS.input)
    # Build the vocabulary.
    line_count = 0
    with open(FLAGS.input, 'r') as input_fd:
        while len(vocab) <= FLAGS.size:
            line = next(input_fd, None)
            if line == None:
                break
            line_count = line_count + 1
            key, value = line.split()
            key = bytes.fromhex(key)
            value = int(value)
            vocab[key] = value
            left_count[key] = value
            right_count[key] = value
            logging.info('New key %s with value %d, vocab size %d, line %d.',
                         key, value, len(vocab), line_count)
            if len(key) > 1:
                left_subkey = key[0:-1]
                left_count[left_subkey] = left_count[left_subkey] - value
                if (len(left_subkey) > 1 and left_count[left_subkey] == 0 and
                    left_subkey in vocab):
                    del vocab[left_subkey]
                    logging.info('Remove left subkey %s, vocab size %d.',
                                 left_subkey, len(vocab))
                right_subkey = key[1:len(key)]
                right_count[right_subkey] = right_count[right_subkey] - value
                if (len(right_subkey) > 1 and right_count[right_subkey] == 0 and
                    right_subkey in vocab):
                    del vocab[right_subkey]
                    logging.info('Remove right subkey %s, vocab size %d.',
                                 right_subkey, len(vocab))
        # 2 more keys can still be inserted if the next key removes 2 subkeys.
        if len(vocab) == FLAGS.size + 1:
            line = next(input_fd, None)
            if line != None:
                line_count = line_count + 1
                key, value = line.split()
                key = bytes.fromhex(key)
                value = int(value)
                vocab[key] = value
                left_count[key] = value
                right_count[key] = value
                logging.info(
                    'New key %s with value %d, vocab size %d, line %d.',
                    key, value, len(vocab), line_count)
                if len(key) > 1:
                    left_subkey = key[0:-1]
                    left_count[left_subkey] = left_count[left_subkey] - value
                    right_subkey = key[1:len(key)]
                    right_count[right_subkey] = (
                        right_count[right_subkey] - value)
                    if (len(left_subkey) > 1 and
                        left_count[left_subkey] == 0 and
                        left_subkey in vocab and len(right_subkey) > 1 and
                        right_count[right_subkey] == 0 and
                        right_subkey in vocab):
                        del vocab[left_subkey]
                        logging.info('Remove left subkey %s, vocab size %d.',
                                     left_subkey, len(vocab))
                        del vocab[right_subkey]
                        logging.info('Remove right subkey %s, vocab size %d.',
                                     right_subkey, len(vocab))            
    # Write vocabulary to file. Python 3.7+ guarantees insertion order for dict.
    logging.info('Write vocabulary of size %d to %s.',
                 min(FLAGS.size, len(vocab)), FLAGS.output)
    with open(FLAGS.output, 'w') as output_fd:
        keys = list(vocab)
        for i in range(min(FLAGS.size, len(vocab))):
            output_fd.write('{} {}\n'.format(keys[i], vocab[keys[i]]))


if __name__ == '__main__':
    app.run(main)
