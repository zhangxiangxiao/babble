"""Reduce sorted gram tree."""

from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input', 'data/tinyshakespeare/train_gram_count_combined.txt',
    'Input file.')
flags.DEFINE_string(
    'output', 'data/tinyshakespeare/train_gram_count_reduced.txt',
    'Output reduced file.')


def main(unused_argv):
    keys = []
    values = []
    logging.info('Read from %s and write to %s.', FLAGS.input, FLAGS.output)
    with open(FLAGS.input, 'r') as input_fd, open(
            FLAGS.output, 'w') as output_fd:
        line_count = 0
        for line in input_fd:
            if line_count % 1000000 == 0:
                logging.info('Process line %d.', line_count)
            line_count = line_count + 1
            key, value = line.split()
            key = bytes.fromhex(key)
            value = int(value)
            if len(key) == len(keys) + 1:
                keys.append(key)
                values.append(value)
            else:
                for i in range(len(key) - 1, len(keys)):
                    if values[i] > 0:
                        output_fd.write('{} {}\n'.format(
                            keys[i].hex(), values[i]))
                keys = keys[0:len(key)]
                values = values[0:len(key)]
                keys[len(key) - 1] = key
                values[len(key) - 1] = value
            if len(values) > 1:
                values[-2] = values[-2] - value
        logging.info('Processing line %d.', line_count)


if __name__ == '__main__':
    app.run(main)
