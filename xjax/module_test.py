"""Unit tests for module."""

from module import Residual, ResLinear, ResConv, ResDeconv

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.random as jrand
from xjax import xnn
from xjax import xrand


class ResidualTest(absltest.TestCase):
    def setUp(self):
        self.linear1 = xnn.Linear(8, 16)
        self.linear2 = xnn.Linear(16, 4)
        self.transfer = xnn.ReLU()
        self.resize = xnn.ResizeLike()
        self.module = Residual(self.linear1, self.linear2)

    def test_forward(self):
        forward, params, states = self.module
        inputs = jrand.normal(xrand.split(), (8,))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((4,), outputs.shape)
        forward1, params1, states1 = self.linear1
        outputs1, states1 = forward1(params1, inputs, states1)
        forward2, params2, states2 = self.transfer
        outputs2, states2 = forward2(params2, outputs1, states2)
        forward3, params3, states3 = self.linear2
        outputs3, states3 = forward3(params3, outputs2, states3)
        forward4, params4, states4 = self.resize
        outputs4, states4 = forward4(params4, [inputs, outputs3], states4)
        ref_outputs = jnp.add(*outputs4)
        self.assertTrue(jnp.allclose(ref_outputs, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs = jrand.normal(xrand.split(), (2, 8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 4), outputs.shape)


class ResLinearTest(absltest.TestCase):
    def setUp(self):
        self.module = ResLinear(8, 16, 4)

    def test_forward(self):
        forward, params, states = self.module
        inputs = jrand.normal(xrand.split(), (8,))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((4,), outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs = jrand.normal(xrand.split(), (2, 8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 4), outputs.shape)


class ResConvTest(absltest.TestCase):
    def setUp(self):
        self.module = ResConv(4, 8, 2, (3,), (3,))

    def test_forward(self):
        forward, params, states = self.module
        inputs = jrand.normal(xrand.split(), (4, 16))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 16), outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs = jrand.normal(xrand.split(), (2, 4, 16))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 2, 16), outputs.shape)


class ResDeconvTest(absltest.TestCase):
    def setUp(self):
        self.module = ResDeconv(4, 8, 2, (3,), (3,), (2,))

    def test_forward(self):
        forward, params, states = self.module
        inputs = jrand.normal(xrand.split(), (4, 16))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 32), outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs = jrand.normal(xrand.split(), (2, 4, 16))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 2, 32), outputs.shape)


if __name__ == '__main__':
    absltest.main()
