"""Combine multiple byte-level grams to one."""

import glob
import queue

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


# Set buffer size to increate file throughput.
BUFFER_SIZE = 1024*1024


def main(argv):
    logging.info('Load input from prefix %s.', FLAGS.input_prefix)
    input_files = glob.glob(FLAGS.input_prefix + '*')
    input_fds = []
    for input_file in input_files:
        logging.info('Open file %s.', input_file)
        input_fds.append(open(input_file, 'r', buffering=BUFFER_SIZE))
    pqueue = queue.PriorityQueue()
    # Initialize input keys and values
    for i in range(len(input_fds)):
        input_fd = input_fds[i]
        input_line = next(input_fd, None)
        if input_line != None:
            input_key, input_value = input_line.split()
            input_value = int(input_value)
            pqueue.put((input_key, input_value, i))
        else:
            input_fd.close()
    logging.info('Write output to %s.', FLAGS.output)
    with open(FLAGS.output, 'w') as output_fd:
        processed_keys = 0
        if not pqueue.empty():
            output_key, output_value, i = pqueue.get()
            input_fd = input_fds[i]
            input_line = next(input_fd, None)
            if input_line != None:
                input_key, input_value = input_line.split()
                input_value = int(input_value)
                pqueue.put((input_key, input_value, i))
            else:
                input_fd.close()
        while not pqueue.empty():
            input_key, input_value, i = pqueue.get()
            if input_key == output_key:
                output_value = output_value + input_value
            else:
                # Write current key value pair.
                output_fd.write('{} {}\n'.format(output_key, output_value))
                output_key = input_key
                output_value = input_value
                processed_keys = processed_keys + 1
                if processed_keys % 100000 == 0:
                    logging.info('Processed keys %d.', processed_keys)
            input_fd = input_fds[i]
            input_line = next(input_fd, None)
            if input_line != None:
                input_key, input_value = input_line.split()
                input_value = int(input_value)
                pqueue.put((input_key, input_value, i))
            else:
                input_fd.close()
        # Write last output key and value.
        output_fd.write('{} {}\n'.format(output_key, output_value))
        processed_keys = processed_keys + 1
    logging.info('Processed keys %d.', processed_keys)


if __name__ == '__main__':
    app.run(main)
