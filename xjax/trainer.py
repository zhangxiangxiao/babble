"""Training logic for babble"""
import os
import time
import unicodedata

from absl import logging
import jax.numpy as jnp
import numpy as np
from xjax import xdl


def replace_unicode_char(char):
    if (unicodedata.category(char)[0] == 'C' or
        unicodedata.category(char)[0] == 'Z'):
        return ' '
    return char

def array_to_string(inputs):
    return ''.join(replace_unicode_char(
        char) for char in np.array(inputs).astype('uint8').tobytes().decode(
            'utf-8', 'replace')).rstrip()


def Trainer(learner, data_train, data_valid, train_steps, test_steps, epochs,
            interval, checkpoint):
    train, test, states = learner
    record = []
    if not os.path.isdir(checkpoint):
        logging.info('Create checkpoint directory %s.', checkpoint)
        os.makedirs(checkpoint)
    record_checkpoint = os.path.join(checkpoint, 'record')
    states_checkpoint = os.path.join(checkpoint, 'states')
    # Load states from checkpoint if it exists.
    if os.path.isfile(record_checkpoint):
        logging.info('Load record from %s.', record_checkpoint)
        record_fd = open(record_checkpoint, 'rb')
        record = xdl.load(record_fd)
        record_fd.close()
    if os.path.isfile(states_checkpoint):
        logging.info('Load states from %s.', states_checkpoint)
        states_fd = open(states_checkpoint, 'rb')
        states = xdl.load(states_fd)
        states_fd.close()

    def log(msg, step, inputs, net_outputs, loss_outputs, eval_outputs,
            total_loss_outputs, total_eval_outputs):
        logging.info(
            '%s step = %d, ae_loss = (%g, %g), gen_loss= (%g, %g), '
            'disc_loss = (%g, %g), dec_eval = (%g, %g)', msg, step,
            loss_outputs[0], total_loss_outputs[0], loss_outputs[1],
            total_loss_outputs[1], loss_outputs[2], total_loss_outputs[2],
            eval_outputs, total_eval_outputs)
        logging.info('   inputs = %s', array_to_string(jnp.argmax(
            inputs[0][0], axis=0)))
        logging.info('  decoded = %s', array_to_string(jnp.argmax(
            net_outputs[0][0], axis=0)))
        logging.info('generated = %s', array_to_string(jnp.argmax(
            net_outputs[1][0], axis=0)))
    train_callback_time = time.time()
    def train_callback(*args):
        nonlocal train_callback_time
        (step, _, _, inputs, net_outputs, loss_outputs, eval_outputs,
         total_loss_outputs, total_eval_outputs) = args
        if time.time() - train_callback_time > interval:
            log('Train', step, inputs, net_outputs, loss_outputs, eval_outputs,
                total_loss_outputs, total_eval_outputs)
            train_callback_time = time.time()
    test_callback_time = time.time()
    def test_callback(*args):
        nonlocal test_callback_time
        (step, inputs, net_outputs, loss_outputs, eval_outputs,
         total_loss_outputs, total_eval_outputs) = args
        if time.time() - test_callback_time > interval:
            log('Test', step, inputs, net_outputs, loss_outputs, eval_outputs,
                total_loss_outputs, total_eval_outputs)
            test_callback_time = time.time()

    def data_iterator(data, steps):
        for _ in range(steps):
            yield data.get_batch()

    def run():
        nonlocal states, record_fd, states_fd
        start_epoch = len(record)
        end_epoch = len(record) + epochs
        for epoch in range(start_epoch, end_epoch):
            logging.info('Train epoch = %d.', epoch)
            loss_train, eval_train, states = train(
                data_iterator(data_train, train_steps), states, train_callback)
            logging.info('Test on train data, epoch = %d', epoch)
            loss_on_train, eval_on_train, states = test(
                data_iterator(data_train, test_steps), states, test_callback)
            logging.info('Test on valid data, epoch = %d', epoch)
            loss_on_valid, eval_on_valid, states = test(
                data_iterator(data_valid, test_steps), states, test_callback)
            logging.info(
                'Finish epoch = %d, ae_loss = (%g, %g, %g), '
                'gen_loss = (%g, %g, %g), disc_loss = (%g, %g, %g), '
                'dec_eval = (%g, %g, %g).', epoch,
                loss_train[0], loss_on_train[0], loss_on_valid[0],
                loss_train[1], loss_on_train[1], loss_on_valid[1],
                loss_train[2], loss_on_train[2], loss_on_valid[2], eval_train,
                eval_on_train, eval_on_valid)
            logging.info('Save to %s', checkpoint)
            record.append((loss_train, eval_train, loss_on_train, eval_on_train,
                           loss_on_valid, eval_on_valid))
            record_fd = open(record_checkpoint, 'wb')
            xdl.dump(record, record_fd)
            record_fd.close()
            states_fd = open(states_checkpoint, 'wb')
            xdl.dump(states, states_fd)
            states_fd.close()

    return run
