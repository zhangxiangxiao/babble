"""Count byte-level grams using Apache Beam."""

import multiprocessing as mp

from absl import app
from absl import flags
from absl import logging
import h5py


FLAGS = flags.FLAGS
flags.DEFINE_string('input', 'data/tinyshakespeare/train.h5',
                    'Input HDF5 file.')
flags.DEFINE_string(
    'output', 'data/tinyshakespeare/train_gram_count',
    'Output count file. Will add postfixes like .0.0, .0.1 etc.')
flags.DEFINE_integer('max_gram_size', 16, 'Maximum byte gram size.')
flags.DEFINE_integer('memory_limit', 17408, 'Limit on memory usage in MB '
                     'before splitting outputs.')
flags.DEFINE_integer('process_size', 16, 'Number of parallel processes.')


def count_grams(master_queue, process_id, process_size, input_file,
                output_prefix, max_gram_size, memory_limit):
    import gc
    import sys
    import h5py
    memory_limit = memory_limit * 1024 * 1024
    sample_count = 0
    file_count = 0
    gram_count = {}
    master_queue.put({
        'rpc': 'logging_info',
        'args': ('Process %d, read from input %s', process_id, input_file),
        'kwargs': {}})
    with h5py.File(input_file, 'r') as input_fd:
        index = input_fd['index']
        content = input_fd['content']
        for i in range(process_id, index.shape[0], process_size):
            if sample_count % 10000 == 0:
                master_queue.put({
                    'rpc': 'logging_info',
                    'args': ('Process %d, sample %d/%d, dictionary %d/%d',
                             process_id, i, index.shape[0],
                             sys.getsizeof(gram_count), memory_limit),
                    'kwargs': {}})
            sample_count = sample_count + 1
            sample_index = index[i, 0]
            sample_length = index[i, 1]
            sample_bytes = content[sample_index:(sample_index + sample_length)]
            sample_bytes = sample_bytes.astype('uint8').tobytes()
            for j in range(len(sample_bytes)):
                for k in range(max_gram_size):
                    if j + k < len(sample_bytes):
                        sample_gram = sample_bytes[j:k+1]
                        gram_count[sample_gram] = gram_count.get(
                            sample_gram, 0) + 1
            if sys.getsizeof(gram_count) > memory_limit:
                master_queue.put({
                    'rpc': 'logging_info',
                    'args': ('Process %d, maximum memory limit %d reached.',
                             process_id, memory_limit),
                    'kwargs': {}})
                output_file = output_prefix + '.{}.{}'.format(
                    process_id, file_count)
                file_count = file_count + 1
                master_queue.put({
                    'rpc': 'logging_info',
                    'args': ('Process %d, save split to %s.', process_id,
                             output_file),
                    'kwargs': {}})
                with open(output_file, 'w') as output_fd:
                    for key in gram_count:
                        value = gram_count[key]
                        output_fd.write('{}, {}\n'.format(key.hex(), value))
                gram_count = {}
                gc.collect()
    master_queue.put({
        'rpc': 'logging_info',
        'args': ('Process %d, sample %d/%d, dictionary %d/%d',
                 process_id, index.shape[0], index.shape[0],
                 sys.getsizeof(gram_count), memory_limit),
        'kwargs': {}})
    if len(gram_count) > 0:
        output_file = output_prefix + '.{}.{}'.format(
            process_id, file_count)
        file_count = file_count + 1
        master_queue.put({
            'rpc': 'logging_info',
            'args': ('Process %d, save split to %s.', process_id, output_file),
            'kwargs': {}})
        with open(output_file, 'w') as output_fd:
            for key in gram_count:
                value = gram_count[key]
                output_fd.write('{}, {}\n'.format(key.hex(), value))
        gram_count = {}
        gc.collect()
    master_queue.put({'rpc': 'exit', 'args': (process_id,), 'kwargs': {}})


def main(argv):
    logging.info('Process master, load input from %s.', FLAGS.input)
    sample_size = None
    with h5py.File(FLAGS.input, 'r') as input_fd:
        sample_size = input_fd['index'].shape[0]
    master_queue = mp.Queue()
    # Start the processes.
    processes = []
    for i in range(FLAGS.process_size):
        process = mp.Process(target=count_grams, args=(
            master_queue, i, FLAGS.process_size, FLAGS.input, FLAGS.output,
            FLAGS.max_gram_size, FLAGS.memory_limit))
        process.start()
        processes.append(process)
    process_count = FLAGS.process_size
    # Process RPCs.
    while process_count > 0:
        rpc = master_queue.get()
        if rpc['rpc'] == 'logging_info':
            logging.info(*rpc['args'], **rpc['kwargs'])
        elif rpc['rpc'] == 'exit':
            logging.info('Process master, process %d has exited.', *rpc['args'])
            process_count = process_count - 1
    # Sync all processes.
    for process in processes:
        process.join()


if __name__ == '__main__':
    app.run(main)
