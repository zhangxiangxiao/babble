"""Unit tests for module."""

from module import Residual, ResLinear, ResConv, ResDeconv
from module import Encoder, Decoder, Discriminator
from module import FeatureInjector, InputInjector, FeatureRandom, InputRandom
from module import AELoss, GenLoss, DiscLoss, DiscLossSigmoid

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
        self.transfer = xnn.Tanh()
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
        forward, params, states = xnn.vmap(self.module)
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
        forward, params, states = xnn.vmap(self.module)
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
        forward, params, states = xnn.vmap(self.module)
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
        forward, params, states = xnn.vmap(self.module)
        inputs = jrand.normal(xrand.split(), (2, 4, 16))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 2, 32), outputs.shape)


class EncoderTest(absltest.TestCase):
    def setUp(self):
        self.module = Encoder(2, 2, 4, 8, 2)

    def test_forward(self):
        forward, params, states = self.module
        inputs = jrand.normal(xrand.split(), (4, 16))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 4), outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module)
        inputs = jrand.normal(xrand.split(), (2, 4, 16))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 2, 4), outputs.shape)


class DecoderTest(absltest.TestCase):
    def setUp(self):
        self.module = Decoder(2, 2, 4, 8, 2)

    def test_forward(self):
        forward, params, states = self.module
        inputs = jrand.normal(xrand.split(), (4, 4))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 16), outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module)
        inputs = jrand.normal(xrand.split(), (2, 4, 4))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 2, 16), outputs.shape)


class DiscriminatorTest(absltest.TestCase):
    def setUp(self):
        self.module = Discriminator(2, 2, 4, 8, 2)

    def test_forward(self):
        forward, params, states = self.module
        inputs = jrand.normal(xrand.split(), (4, 16))
        outputs, states = forward(params, inputs, states)
        self.assertEqual(6, len(outputs))
        self.assertEqual((8, 16), outputs[0].shape)
        self.assertEqual((8, 16), outputs[1].shape)
        self.assertEqual((8, 8), outputs[2].shape)
        self.assertEqual((8, 8), outputs[3].shape)
        self.assertEqual((8, 4), outputs[4].shape)
        self.assertEqual((2, 4), outputs[5].shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module)
        inputs = jrand.normal(xrand.split(), (2, 4, 16))
        outputs, states = forward(params, inputs, states)
        self.assertEqual(6, len(outputs))
        self.assertEqual((2, 8, 16), outputs[0].shape)
        self.assertEqual((2, 8, 16), outputs[1].shape)
        self.assertEqual((2, 8, 8), outputs[2].shape)
        self.assertEqual((2, 8, 8), outputs[3].shape)
        self.assertEqual((2, 8, 4), outputs[4].shape)
        self.assertEqual((2, 2, 4), outputs[5].shape)


class FeatureInjectorTest(absltest.TestCase):
    def setUp(self):
        self.module = FeatureInjector(1e-1)

    def test_forward(self):
        forward, params, states = self.module
        inputs = jrand.normal(xrand.split(), (4, 8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module)
        inputs = jrand.normal(xrand.split(), (2, 4, 8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual(inputs.shape, outputs.shape)


class InputInjectorTest(absltest.TestCase):
    def setUp(self):
        self.module = InputInjector(4, 1e-1)

    def test_forward(self):
        forward, params, states = self.module
        inputs = jrand.normal(xrand.split(), (4, 8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module)
        inputs = jrand.normal(xrand.split(), (2, 4, 8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual(inputs.shape, outputs.shape)


class FeatureRandomTest(absltest.TestCase):
    def setUp(self):
        self.module = FeatureRandom()

    def test_forward(self):
        forward, params, states = self.module
        inputs = jrand.normal(xrand.split(), (4, 8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module)
        inputs = jrand.normal(xrand.split(), (2, 4, 8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual(inputs.shape, outputs.shape)


class InputRandomTest(absltest.TestCase):
    def setUp(self):
        self.module = InputRandom(4)

    def test_forward(self):
        forward, params, states = self.module
        inputs = jrand.normal(xrand.split(), (4, 8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module)
        inputs = jrand.normal(xrand.split(), (2, 4, 8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual(inputs.shape, outputs.shape)


class AELossTest(absltest.TestCase):
    def setUp(self):
        self.module = AELoss()

    def test_forward(self):
        forward, params, states = self.module
        inputs = [jrand.normal(xrand.split(), (4, 8)),
                  jrand.normal(xrand.split(), (4, 8)),
                  jrand.normal(xrand.split(), (8, ))]
        outputs, states = forward(params, inputs, states)
        self.assertEqual((), outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module)
        inputs = [jrand.normal(xrand.split(), (2, 4, 8)),
                  jrand.normal(xrand.split(), (2, 4, 8)),
                  jrand.normal(xrand.split(), (2, 8))]
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2,), outputs.shape)


class GenLossTest(absltest.TestCase):
    def setUp(self):
        self.module = GenLoss()

    def test_forward(self):
        forward, params, states = self.module
        inputs = [[jrand.normal(xrand.split(), (4, 8)),
                   jrand.normal(xrand.split(), (4, 4))],
                  [jrand.normal(xrand.split(), (4, 8)),
                   jrand.normal(xrand.split(), (4, 4))]]
        outputs, states = forward(params, inputs, states)
        self.assertEqual((), outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module)
        inputs = [[jrand.normal(xrand.split(), (2, 4, 8)),
                   jrand.normal(xrand.split(), (2, 4, 4))],
                  [jrand.normal(xrand.split(), (2, 4, 8)),
                   jrand.normal(xrand.split(), (2, 4, 4))]]
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2,), outputs.shape)


class DiscLossTest(absltest.TestCase):
    def setUp(self):
        self.module = DiscLoss()

    def test_forward(self):
        forward, params, states = self.module
        inputs = [[jrand.normal(xrand.split(), (4, 8)),
                   jrand.normal(xrand.split(), (4, 4))],
                  [jrand.normal(xrand.split(), (4, 8)),
                   jrand.normal(xrand.split(), (4, 4))]]
        outputs, states = forward(params, inputs, states)
        self.assertEqual((), outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module)
        inputs = [[jrand.normal(xrand.split(), (2, 4, 8)),
                   jrand.normal(xrand.split(), (2, 4, 4))],
                  [jrand.normal(xrand.split(), (2, 4, 8)),
                   jrand.normal(xrand.split(), (2, 4, 4))]]
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2,), outputs.shape)


class DiscLossSigmoidTest(absltest.TestCase):
    def setUp(self):
        self.module = DiscLossSigmoid()

    def test_forward(self):
        forward, params, states = self.module
        inputs = [[jrand.normal(xrand.split(), (4, 8)),
                   jrand.normal(xrand.split(), (4, 4))],
                  [jrand.normal(xrand.split(), (4, 8)),
                   jrand.normal(xrand.split(), (4, 4))]]
        outputs, states = forward(params, inputs, states)
        self.assertEqual((), outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module)
        inputs = [[jrand.normal(xrand.split(), (2, 4, 8)),
                   jrand.normal(xrand.split(), (2, 4, 4))],
                  [jrand.normal(xrand.split(), (2, 4, 8)),
                   jrand.normal(xrand.split(), (2, 4, 4))]]
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2,), outputs.shape)


if __name__ == '__main__':
    absltest.main()
