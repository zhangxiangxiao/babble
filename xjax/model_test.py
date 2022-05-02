"""Unit tests for model module."""

from model import ATNNFAE

import math

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.random as jrand
from xjax import xmod
from xjax import xnn
from xjax import xrand


class ATNNFAETest(absltest.TestCase):
    def setUp(self):
        # Encoder is a 2-layer MLP.
        self.enc = xnn.Sequential(
            xnn.Linear(8, 16), xnn.ReLU(), xnn.Linear(16, 4), xnn.Standardize())
        # Decoder / generator is a 2-layer MLP.
        self.dec = xnn.Sequential(
            xnn.Standardize(), xnn.Linear(4, 16), xnn.ReLU(), xnn.Linear(16, 8))
        # Discriminator is a 2-layer MLP.
        self.disc = xnn.Sequential(
            xnn.Linear(8, 16), xnn.ReLU(), xnn.Linear(16, 1), xnn.Mean())
        # Noise injector adds noise from normal distribution.
        self.inj = xnn.Sequential(
            # inputs -> [inputs] -> [inputs, inputs]
            xnn.Pack(), xnn.Group([0, 0]),
            # [inputs, inputs] -> [inputs, noise]
            xnn.Parallel(xnn.Identity(),
                         xnn.Sequential(xnn.NormalLike(), xnn.MulConst(0.1))),
            # [inputs, noise] -> [inputs + noise]
            xnn.Add())
        # Random generator.
        self.rnd = xnn.NormalLike()
        # Auto-encoder loss is square loss.
        self.ae_loss = xnn.Sequential(xnn.Subtract(), xnn.Square(), xnn.Sum())
        # Generator loss is log-cosh loss towards 0.
        self.gen_loss = xnn.Sequential(
            # inputs -> [inputs] -> [inputs, inputs]
            xnn.Pack(), xnn.Group([0, 0]),
            # [inputs, inputs] -> [inputs, zeros]
            xnn.Parallel(xnn.Identity(), xnn.ZerosLike()),
            # [inputs, zeros] -> logcosh
            xnn.LogCosh(), xnn.Mean())
        # Discriminator is log-cosh loss towards 0 for real, and -1 for fake.
        self.disc_loss = xnn.Sequential(
            # [real, fake] -> [[real, real], [fake, fake]]
            xnn.Group([[0, 0], [1, 1]]),
            # [[real, real], [fake, fake]] -> [real_logcosh, fake_logcosh]
            xnn.Parallel(
                # [real, real] -> real_logcosh
                xnn.Sequential(
                    # [real, real] -> [real, zeros]
                    xnn.Parallel(xnn.Identity(), xnn.ZerosLike()),
                    # [real, zeros] -> real_logcosh
                    xnn.LogCosh()),
                # [fake, fake] -> fake_logcosh
                xnn.Sequential(
                    # [fake, fake] -> [fake, ones]
                    xnn.Parallel(xnn.Identity(), xnn.OnesLike()),
                    # [fake, ones] -> fake_logcosh
                    xnn.LogCosh())),
            # [real_logcosh, fake_logcosh] -> loss
            xnn.Add(), xnn.Mean())
        # Build the model
        self.model = ATNNFAE(self.enc, self.dec, self.disc, self.inj, self.rnd,
                             self.ae_loss, self.gen_loss, self.disc_loss)
        # Inputs is a vector of size 8.
        self.inputs = jrand.normal(xrand.split(), shape=(8,))


    def test_forward(self):
        forward, _, params, states = self.model
        inputs = self.inputs
        net_outputs, loss_outputs, states = forward(params, inputs, states)
        dec_outputs, gen_outputs, disc_outputs = net_outputs
        real_outputs, fake_outputs = disc_outputs
        ae_loss_outputs, gen_loss_outputs, disc_loss_outputs = loss_outputs
        enc_forward, enc_params, enc_states = self.enc
        enc_outputs, enc_states = enc_forward(enc_params, inputs, enc_states)
        inj_forward, inj_params, inj_states = self.inj
        inj_outputs, inj_states = inj_forward(
            inj_params, enc_outputs, inj_states)
        dec_forward, dec_params, dec_states = self.dec
        ref_dec_outputs, dec_states = dec_forward(
            dec_params, inj_outputs, dec_states)
        self.assertTrue(jnp.allclose(ref_dec_outputs, dec_outputs))
        rnd_forward, rnd_params, rnd_states = self.rnd
        rnd_outputs, rng_states = rnd_forward(
            rnd_params, enc_outputs, rnd_states)
        ref_gen_outputs, gen_states = dec_forward(
            dec_params, rnd_outputs, dec_states)
        self.assertTrue(jnp.allclose(ref_gen_outputs, gen_outputs))
        disc_forward, disc_params, disc_states = self.disc
        ref_real_outputs, disc_states = disc_forward(
            disc_params, dec_outputs, disc_states)
        self.assertTrue(jnp.allclose(ref_real_outputs, real_outputs))
        ref_fake_outputs, disc_states = disc_forward(
            disc_params, gen_outputs, disc_states)
        self.assertTrue(jnp.allclose(ref_fake_outputs, fake_outputs))
        ae_loss_forward, ae_loss_params, ae_loss_states = self.ae_loss
        ref_ae_loss_outputs, ae_loss_states = ae_loss_forward(
            ae_loss_params, [dec_outputs, inputs], ae_loss_states)
        self.assertTrue(jnp.allclose(ref_ae_loss_outputs, ae_loss_outputs))
        gen_loss_forward, gen_loss_params, gen_loss_states = self.gen_loss
        ref_gen_loss_outputs, gen_loss_states = gen_loss_forward(
            gen_loss_params, fake_outputs, ae_loss_states)
        self.assertTrue(jnp.allclose(ref_gen_loss_outputs, gen_loss_outputs))
        disc_loss_forward, disc_loss_params, disc_loss_states = self.disc_loss
        ref_disc_loss_outputs, disc_loss_states = disc_loss_forward(
            disc_loss_params, disc_outputs, disc_loss_states)
        self.assertTrue(jnp.allclose(ref_disc_loss_outputs, disc_loss_outputs))

    def test_backward(self):
        forward, backward, params, states = self.model
        inputs = self.inputs
        grads, net_outputs, loss_outputs, _ = backward(
            params, inputs, states)
        ref_net_outputs, ref_loss_outputs, _ = forward(
            params, inputs, states)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     [ref_net_outputs, ref_loss_outputs],
                     [net_outputs, loss_outputs])
        def ref_forward_enc(params, inputs, states):
            _, loss_outputs, _ = forward(params, inputs, states)
            return loss_outputs[0]
        ref_backward_enc = jax.grad(ref_forward_enc)
        ref_grads = ref_backward_enc(params, inputs, states)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     ref_grads[0], grads[0])
        def ref_forward_dec(params, inputs, states):
            _, loss_outputs, _ = forward(params, inputs, states)
            return loss_outputs[0] + loss_outputs[1]
        ref_backward_dec = jax.grad(ref_forward_dec)
        ref_grads = ref_backward_dec(params, inputs, states)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     ref_grads[1], grads[1])
        def ref_forward_disc(params, inputs, states):
            _, loss_outputs, _ = forward(params, inputs, states)
            return loss_outputs[2]
        ref_backward_disc = jax.grad(ref_forward_disc)
        ref_grads = ref_backward_disc(params, inputs, states)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     ref_grads[2], grads[2])

    def test_vmap(self):
        forward, backward, params, states = xmod.vmap(self.model, 2)
        inputs = jrand.normal(xrand.split(), (2, 8))
        net_outputs, loss_outputs, states = forward(params, inputs, states)
        dec_outputs, gen_outputs, disc_outputs = net_outputs
        real_outputs, fake_outputs = disc_outputs
        ae_loss_outputs, gen_loss_outputs, disc_loss_outputs = loss_outputs
        self.assertEqual((2, 8), dec_outputs.shape)
        self.assertEqual((2, 8), gen_outputs.shape)
        self.assertEqual((2,), real_outputs.shape)
        self.assertEqual((2,), fake_outputs.shape)
        self.assertEqual((2,), ae_loss_outputs.shape)
        self.assertEqual((2,), gen_loss_outputs.shape)
        self.assertEqual((2,), disc_loss_outputs.shape)
        grads, net_outputs, loss_outputs, states = backward(
            params, inputs, states)
        jax.tree_map(lambda p, g: self.assertEqual((2,) + p.shape, g.shape),
                     params, grads)


if __name__ == '__main__':
    absltest.main()
