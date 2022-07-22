"""Create hex data."""

import h5py
from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS
flags.DEFINE_string('input', 'data/tinyshakespeare/train.h5',
                    'Input HDF5 file.')
flags.DEFINE_string('output', 'data/tinyshakespeare/train.hex',
                    'Output HEX file.')
flags.DEFINE_boolean('cache', False, 'Whether to cache the input in memory.')


def main(unused_argv):
    logging.info('Load input from %s.', FLAGS.input)
    with h5py.File(FLAGS.input, 'r') as input_fd:
        index = input_fd['index']
        content = input_fd['content']
        if FLAGS.cache:
            logging.info('Cache the input in memory.')
            index = index[:]
            content = content[:]
        logging.info('Save output to %s.', FLAGS.output)
        with open(FLAGS.output, 'w') as output_fd:
            for i in range(index.shape[0]):
                if i % 100000 == 0:
                    logging.info('Process sample %d/%d.', i, index.shape[0])
                content_index = index[i, 0]
                content_length = index[i, 1]
                sample_bytes = content[content_index:(
                    content_index + content_length)]
                sample_hex = sample_bytes.astype('uint8').tobytes().hex()
                output_fd.write(sample_hex)
                output_fd.write('\n')
            logging.info('Process sample %d/%d.',
                         index.shape[0], index.shape[0])


if __name__ == '__main__':
    app.run(main)
