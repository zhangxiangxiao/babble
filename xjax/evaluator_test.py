"""Unit test for evaluator."""

from evaluator import Evaluator

from absl.testing import absltest
import jax.numpy as jnp
import jax.random as jrand
from xjax import xrand
from xjax import xeval


class EvaluatorTest(absltest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator()

    def test_evaluate(self):
        evaluate, states = self.evaluator
        inputs = [jrand.normal(xrand.split(), (256, 16)),
                  jrand.normal(xrand.split(), (16,))]
        net_outputs = [jrand.normal(xrand.split(), (256, 16)),
                       jrand.normal(xrand.split(), (256, 16)),
                       jrand.normal(xrand.split(), (256, 16)),
                       jrand.normal(xrand.split()), jrand.normal(xrand.split())]
        outputs, states = evaluate(inputs, net_outputs, states)
        self.assertEqual(2, len(outputs))

    def test_vmap(self):
        evaluate, states = xeval.vmap(self.evaluator, 2)
        inputs = [jrand.normal(xrand.split(), (2, 256, 16)),
                  jrand.normal(xrand.split(), (2, 16))]
        net_outputs = [jrand.normal(xrand.split(), (2, 256, 16)),
                       jrand.normal(xrand.split(), (2, 256, 16)),
                       jrand.normal(xrand.split(), (2, 256, 16)),
                       jrand.normal(xrand.split(), (2,)),
                       jrand.normal(xrand.split(), (2,))]
        outputs, states = evaluate(inputs, net_outputs, states)
        self.assertEqual(2, len(outputs))


if __name__ == '__main__':
    absltest.main()
