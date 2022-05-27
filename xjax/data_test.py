"""Unit test for Data class"""

from data import Data

from absl.testing import absltest
import numpy as np


class DataTest(absltest.TestCase):
    def test_get_batch(self):
        data = Data(filename='data/obama/train.h5', batch=4, min_len=4,
                    max_len=64, cache=True)
        for i in range(4):
            inputs, weights = data.get_batch()
            self.assertEqual(3, inputs.ndim)
            self.assertEqual(2, weights.ndim)
            self.assertEqual(0, inputs.shape[0] % 4)
            self.assertEqual(0, weights.shape[0] % 4)
            self.assertEqual(256, inputs.shape[1])
            self.assertEqual(0, weights.shape[1] % 4)
            self.assertEqual(0, inputs.shape[2] % 4)


if __name__ == '__main__':
    absltest.main()
