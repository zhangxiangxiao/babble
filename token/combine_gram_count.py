"""Combine multiple byte-level grams to one."""

import glob

from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input_prefix', 'data/tinyshakespeare/train_gram_count_sorted.',
    'Input file prefix.')
flags.DEFINE_string(
    'output', 'data/tinyshakespeare/train_gram_count_combined.txt',
    'Output file.')


def main(argv):
    logging.info('Load input from prefix %s.', FLAGS.input_prefix)
    input_files = glob.glob(FLAGS.input_prefix + '*')
    input_fds = []
    for input_file in input_files:
        logging.info('Open file %s.', input_file)
        input_fds.append(open(input_file, 'r'))
    # Initialize input keys and values
    input_keys = []
    input_values = []
    closed_files = len(input_files)
    for input_fd in input_fds:
        input_line = next(input_fd, None)
        if input_line != None:
            input_key, input_value = input_line.split()
            input_keys.append(input_key)
            input_values.append(int(input_value))
        else:
            input_keys.append(None)
            input_values.append(None)
            input_fd.close()
            closed_files = closed_files - 1
    logging.info('Write output to %s.', FLAGS.output)
    with open(FLAGS.output, 'w') as output_fd:
        processed_keys = 0
        while closed_files > 0:
            if processed_keys % 100000 == 0:
                logging.info('Processed keys %d.', processed_keys)
            output_key = None
            output_value = None
            # Find the smallest key in currently opened keys.
            for i in range(len(input_fds)):
                if input_keys[i] == None:
                    continue
                if output_key == None or output_key > input_keys[i]:
                    output_key = input_keys[i]
                    output_value = input_values[i]
                elif output_key == input_keys[i]:
                    output_value = output_value + input_values[i]
            # Write current key value pair
            output_fd.write('{} {}\n'.format(output_key, output_value))
            processed_keys = processed_keys + 1
            # Progress for the currently opened files.
            for i in range(len(input_fds)):
                if input_keys[i] == None:
                    continue
                if input_keys[i] == output_key:
                    input_line = next(input_fds[i], None)
                    if input_line != None:
                        input_key, input_value = input_line.split()
                        input_keys[i] = input_key
                        input_values[i] = int(input_value)
                    else:
                        input_keys[i] = None
                        input_values[i] = None
                        input_fds[i].close()
                        closed_files = closed_files - 1
    logging.info('Processed keys %d.', processed_keys)


if __name__ == '__main__':
    app.run(main)
