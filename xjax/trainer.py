"""Training logic for babble"""
import os
import time

from absl import logging
import jax.numpy as jnp
from xjax import xdl

def array_to_string(inputs):
    return ''

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
            'disc_loss = (%g, %g), tar_eval = (%g, %g), '
            'dec_eval = (%g, %g)', msg, step, jnp.mean(loss_outputs[0]),
            jnp.mean(total_loss_outputs[0]), jnp.mean(loss_outputs[1]),
            jnp.mean(total_loss_outputs[1]), jnp.mean(loss_outputs[2]),
            jnp.mean(total_loss_outputs[2]), jnp.mean(eval_outputs[0]),
            jnp.mean(total_eval_outputs[0]), jnp.mean(eval_outputs[1]),
            jnp.mean(total_eval_outputs[1]))
        logging.info('   inputs = %s', array_to_string(inputs[0]))
        logging.info('  targets = %s', array_to_string(net_outputs[0]))
        logging.info('  decoded = %s', array_to_string(net_outputs[1]))
        logging.info('generated = %s', array_to_string(net_outputs[2]))
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
                'tar_eval = (%g, %g, %g), dec_eval = (%g, %g, %g).',
                epoch, jnp.mean(loss_train[0]), jnp.mean(loss_on_train[0]),
                jnp.mean(loss_on_valid[0]), jnp.mean(loss_train[1]),
                jnp.mean(loss_on_train[1]), jnp.mean(loss_on_valid[1]),
                jnp.mean(loss_train[2]), jnp.mean(loss_on_train[2]),
                jnp.mean(loss_on_valid[2]), jnp.mean(eval_train[0]),
                jnp.mean(eval_on_train[0]), jnp.mean(eval_on_valid[0]),
                jnp.mean(eval_train[1]), jnp.mean(eval_on_train[1]),
                jnp.mean(eval_on_valid[1]))
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
