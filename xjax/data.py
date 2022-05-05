""" Data class."""

import h5py
import jax.nn as jnn
import jax.numpy as jnp
import numpy
import random


class Data:
    def __init__(self, filename, batch=16, step=16, min_len=1, max_len=256,
                 cache=True):
        self.filename = filename
        self.batch = batch
        self.step = step
        self.min_len = min_len
        self.max_len = max_len

        self.data_fd = h5py.File(filename, 'r')
        self.index = self.data_fd['index']
        self.length = self.data_fd['length']
        self.content = self.data_fd['content']

        self.cache = cache

    @property
    def cache(self):
        return self.data_fd is None

    @cache.setter
    def cache(self, cache):
        if cache and self.data_fd is not None:
            self.index = self.index[:]
            self.length = self.length[:]
            self.content = self.content[:]
            self.data_fd.close()
            self.data_fd = None
        elif not cache and self.data_fd is None:
            self.data_fd = h5py.File(filename, 'r')
            self.index = self.data_fd['index']
            self.length = self.data_fd['length']
            self.content = self.data_fd['content']

    def get_batch(self):
        # Get one sample
        sample_index = random.randrange(
            self.length[self.min_len - 1], self.index.shape[0])
        # Calcuate the lower and upper index
        lower_length = int(
            (self.index[sample_index, 1] - 1) / self.step) * self.step + 1
        upper_length = lower_length + self.step - 1
        if lower_length > self.max_len:
            lower_index = int(self.length[self.max_len])
            upper_index = int(self.index.shape[0])
        else:
            lower_index = self.length[lower_length - 1]
            if upper_length >= self.length.shape[0]:
                upper_index = self.index.shape[0]
            else:
                upper_index = self.length[upper_length - 1]
        # Create batch bytes and batch length
        inputs_length = min(upper_length, self.max_len)
        inputs_bytes = numpy.zeros(
            shape=(self.batch, inputs_length), dtype='uint8')
        inputs_weight = numpy.zeros(shape=(self.batch, inputs_length))
        # Copy the bytes for the first sample
        content_index = int(self.index[sample_index, 0])
        content_length = int(min(self.index[sample_index, 1], inputs_length))
        inputs_bytes[0, 0:content_length] = self.content[
            content_index:(content_index + content_length)]
        inputs_weight[0, 0:content_length] = inputs_length / content_length
        # Copy the bytes for the rest of the samples
        for i in range(1, self.batch):
            sample_index = random.randrange(lower_index, upper_index)
            content_index = int(self.index[sample_index, 0])
            content_length = int(min(
                self.index[sample_index, 1], upper_length, self.max_len))
            padding_length = min(
                (int((content_length - 1) / self.step) + 1) * self.step,
                inputs_length)
            inputs_bytes[i, 0:content_length] = self.content[
                content_index:(content_index + content_length)]
            inputs_weight[i, 0:content_length] = inputs_length / content_length
        inputs = jnp.transpose(jnn.one_hot(inputs_bytes, 256), (0, 2, 1))
        return inputs, inputs_weight
