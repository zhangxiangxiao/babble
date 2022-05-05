"""Unit test for Data class"""

from data import Data

from absl.testing import absltest


class DataTest(absltest.TestCase):
    def test_get_batch(self):
        data = Data(filename='data/obama/train.h5', batch=4, step=4, min_len=1,
                    max_len=64, cache=True)
        for i in range(4):
            inputs_bytes, inputs_weight = data.get_batch()
            self.assertEqual(2, inputs_bytes.ndim)
            self.assertEqual(2, inputs_weight.ndim)
            self.assertEqual(4, inputs_bytes.shape[0])
            self.assertEqual(4, inputs_weight.shape[0])
            self.assertEqual(0, inputs_bytes.shape[1] % 4)
            self.assertEqual(0, inputs_weight.shape[1] % 4)


if __name__ == '__main__':
    absltest.main()
