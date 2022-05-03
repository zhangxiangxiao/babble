"""Adversarially trained normalized Noisy-Feature Auto-Encoder Model"""

import module
import jax.nn.initializers as jinit
from xjax import xnn
from xjax.xmod import ModelTuple
from xjax.xmod import vjp, vjp_full, vjp_inputs, map_ones_like, map_add


def ATNNFAE(enc, dec, disc, inj, rnd, ae_loss, gen_loss, disc_loss):
    """Adversarially-Trained Normalized Noisy-Feature Auto-Encoder model.

    Args:
      enc: the encoder module.
      dec: the decoder / generator module.
      disc: the discriminator module.
      inj: the noise injection modle.
      rnd: the random noise generation module.
      ae_loss: the autoencode loss.
      gen_loss: the generator loss.
      disc_loss: the discriminator loss.

    Returns:
      forward: the forward function that returns net_outputs, loss_outputs, and
        states.
      backward: the backward function that returns grads and forward returns.
      initial_params: the initial parameters from net.
      initial_states: the initial states.
    """
    initial_params = (enc[1], dec[1], disc[1])
    initial_states = (enc[2], dec[2], disc[2], inj[2], rnd[2], ae_loss[2],
                      gen_loss[2], disc_loss[2])
    enc_forward, dec_forward, disc_forward = enc[0], dec[0], disc[0]
    inj_forward, inj_params, _ = inj
    rnd_forward, rnd_params, _ = rnd
    ae_loss_forward, ae_loss_params, _ = ae_loss
    gen_loss_forward, gen_loss_params, _ = gen_loss
    disc_loss_forward, disc_loss_params, _ = disc_loss
    def forward(params, inputs, states):
        enc_params, dec_params, disc_params = params
        enc_states, dec_states, disc_states = states[:3]
        inj_states, rnd_states = states[3:5]
        ae_loss_states, gen_loss_states, disc_loss_states = states[5:]
        enc_outputs, enc_states = enc_forward(enc_params, inputs, enc_states)
        inj_outputs, inj_states = inj_forward(
            inj_params, enc_outputs, inj_states)
        rnd_outputs, rnd_states = rnd_forward(
            rnd_params, enc_outputs, rnd_states)
        dec_outputs, dec_states = dec_forward(
            dec_params, inj_outputs, dec_states)
        gen_outputs, dec_states = dec_forward(
            dec_params, rnd_outputs, dec_states)
        real_outputs, disc_states = disc_forward(
            disc_params, dec_outputs, disc_states)
        fake_outputs, disc_states = disc_forward(
            disc_params, gen_outputs, disc_states)
        disc_outputs = [real_outputs, fake_outputs]
        net_outputs = [dec_outputs, gen_outputs, disc_outputs]
        ae_loss_outputs, ae_loss_states = ae_loss_forward(
            ae_loss_params, [dec_outputs, inputs], ae_loss_states)
        gen_loss_outputs, gen_loss_states = gen_loss_forward(
            gen_loss_params, fake_outputs, gen_loss_states)
        disc_loss_outputs, disc_loss_states = disc_loss_forward(
            disc_loss_params, disc_outputs, disc_loss_states)
        states = (enc_states, dec_states, disc_states, inj_states, rnd_states,
                  ae_loss_states, gen_loss_states, disc_loss_states)
        loss_outputs = [ae_loss_outputs, gen_loss_outputs, disc_loss_outputs]
        return net_outputs, loss_outputs, states
    def backward(params, inputs, states):
        enc_params, dec_params, disc_params = params
        enc_states, dec_states, disc_states = states[:3]
        inj_states, rnd_states = states[3:5]
        ae_loss_states, gen_loss_states, disc_loss_states = states[5:]
        # Forward propagate and build backward graph.
        enc_vjpf, enc_outputs, ens_states = vjp(
            enc_forward, enc_params, inputs, enc_states)
        inj_vjpf, inj_outputs, inj_states = vjp_inputs(
            inj_forward, inj_params, enc_outputs, inj_states)
        rnd_outputs, rnd_states = rnd_forward(
            rnd_params, enc_outputs, rnd_states)
        dec_vjpf, dec_outputs, dec_states = vjp_full(
            dec_forward, dec_params, inj_outputs, dec_states)
        gen_vjpf, gen_outputs, dec_states = vjp(
            dec_forward, dec_params, rnd_outputs, dec_states)
        disc_vjpf_real, real_outputs, disc_states = vjp(
            disc_forward, disc_params, dec_outputs, disc_states)
        disc_vjpf_fake, fake_outputs, disc_states = vjp_full(
            disc_forward, disc_params, gen_outputs, disc_states)
        disc_outputs = [real_outputs, fake_outputs]
        net_outputs = [dec_outputs, gen_outputs, disc_outputs]
        ae_loss_vjpf, ae_loss_outputs, ae_loss_states = vjp_inputs(
            ae_loss_forward, ae_loss_params, [dec_outputs, inputs],
            ae_loss_states)
        gen_loss_vjpf, gen_loss_outputs, gen_loss_states = vjp_inputs(
            gen_loss_forward, gen_loss_params, fake_outputs, gen_loss_states)
        disc_loss_vjpf, disc_loss_outputs, disc_loss_states = vjp_inputs(
            disc_loss_forward, disc_loss_params, disc_outputs, disc_loss_states)
        loss_outputs = [ae_loss_outputs, gen_loss_outputs, disc_loss_outputs]
        states = [enc_states, dec_states, disc_states, inj_states, rnd_states,
                  ae_loss_states, gen_loss_states, disc_loss_states]
        # Backward propagate to autoencoder.
        grads_ae_loss_outputs = map_ones_like(ae_loss_outputs)
        grads_dec_outputs, _ = ae_loss_vjpf(grads_ae_loss_outputs)
        grads_dec_params_ae, grads_inj_outputs = dec_vjpf(grads_dec_outputs)
        grads_enc_outputs = inj_vjpf(grads_inj_outputs)
        grads_enc_params = enc_vjpf(grads_enc_outputs)
        # Backward propagate to generator.
        grads_gen_loss_outputs = map_ones_like(gen_loss_outputs)
        grads_fake_outputs_gen = gen_loss_vjpf(grads_gen_loss_outputs)
        _, grads_gen_outputs = disc_vjpf_fake(grads_fake_outputs_gen)
        grads_dec_params_gen = gen_vjpf(grads_gen_outputs)
        # Backward propagate to discriminator
        grads_disc_loss_outputs = map_ones_like(disc_loss_outputs)
        grads_real_outputs, grads_fake_outputs_disc = (
            disc_loss_vjpf(grads_disc_loss_outputs))
        grads_disc_params_real = disc_vjpf_real(grads_real_outputs)
        grads_disc_params_fake, _ = disc_vjpf_fake(grads_fake_outputs_disc)
        # Add parameters together
        grads_dec_params = map_add(grads_dec_params_ae, grads_dec_params_gen)
        grads_disc_params = map_add(
            grads_disc_params_real, grads_disc_params_fake)
        grads = (grads_enc_params, grads_dec_params, grads_disc_params)
        return grads, net_outputs, loss_outputs, states
    return ModelTuple(forward, backward, initial_params, initial_states)


def Encoder(level, depth, in_dim, feat_dim, out_dim, transfer=xnn.ReLU,
            kernel=(3,), pool=(2,), sigma=1e-6):
    """Encoder."""
    layers = []
    # inputs, shape=(i, l) -> inputs, shape=(l, i)
    layers.append(xnn.Transpose())
    # inputs -> features
    layers.append(module.ResLinear(
        in_dim, feat_dim, transfer, w_init=jinit.normal(sigma),
        b_init=jinit.normal(sigma)))
    # features, shape=(l, f) -> features, shape=(f, l)
    layers.append(xnn.Transpose())
    # features -> features.
    for _ in range(level):
        for _ in range(depth):
            # features -> features
            layers.append(transfer())
            layers.append(module.ResConv(
                feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
                w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
        # features, shape=(f, l) -> features, shape=(f, l/2)
        layers.append(xnn.MaxPool(pool))
    # features, shape=(f, l) -> features, shape=(l, f)
    layers.append(xnn.Transpose())
    # features, shape=(l, f) -> outputs, shape=(l, o)
    layers.append(transfer())
    layers.append(module.ResLinear(
        feat_dim, out_dim, transfer, w_init=jinit.normal(sigma),
        b_init=jinit.normal(sigma)))
    # features -> features
    layers.append(xnn.Standardize(axis=0))
    return xnn.Sequential(*layers)


def Decoder(level, depth, in_dim, feat_dim, out_dim, kernel=(3,), stride=(2,),
            sigma=1e-6):
    """Decoder."""
    layers = []
    # inputs -> inputs
    layers.append(xnn.Standardize(axis=0))
    # inputs, shape=(i, l) -> inputs, shape=(l, i)
    layers.append(xnn.Transpose())
    # inputs -> features
    layers.append(module.ResLinear(
        in_dim, feat_dim, transfer, w_init=jinit.normal(sigma),
        b_init=jinit.normal(sigma)))
    # features, shape=(l, f) -> features, shape=(f, l)
    layers.append(xnn.Transpose())
    # features -> features
    for _ in range(level):
        # features, shape=(f, l) -> features, shape=(f, l*2)
        layers.append(transfer())
        layers.append(module.ResDeconv(
            feat_dim, feat_dim, feat_dim, kernel, kernel, stride,
            transfer=transfer, w_init=jinit.normal(sigma),
            b_init=jinit.normal(sigma)))
        for _ in range(depth - 1):
            # features -> features
            layers.append(transfer())
            layers.append(module.ResConv(
                feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
                w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    # features, shape=(f, l) -> features, shape=(l, f)
    layers.append(xnn.Transpose())
    # features, shape=(l, f) -> outputs, shape=(l, o)
    layers.append(transfer())
    layers.append(module.ResLinear(
        feat_dim, out_dim, transfer, w_init=jinit.normal(sigma),
        b_init=jinit.normal(sigma)))
    return xnn.Sequential(*layers)


def Discriminator(level, depth, in_dim, feat_dim, out_dim, kernel=(3,),
                  pool=(2,), sigma=1e-6):
    """Discriminator that is dense."""
    layers = []
    layers.append(xnn.Sequential(
        # inputs -> inputs, as softmax on log-sigmoid.
        xnn.MulConst(-1), xnn.LogSigmoid(), xnn.Softmax(),
        # inputs, shape=(i, l) -> inputs, shape=(l, i)
        xnn.Transpose(),
        # inputs -> features
        module.ResLinear(
            in_dim, feat_dim, transfer, w_init=jinit.normal(sigma),
            b_init=jinit.normal(sigma)),
        # features, shape=(l, f) -> features, shape=(f, l)
        xnn.Transpose()))
    # features -> features.
    for _ in range(level):
        for _ in range(depth - 1):
            # features -> features
            layers.append(xnn.Sequential(
                transfer(),
                module.ResConv(
                    feat_dim, feat_dim, feat_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
        # features, shape=(f, l) -> features, shape=(f, l/2)
        layers.append(xnn.Sequential(
            transfer(),
            module.ResConv(
                feat_dim, feat_dim, feat_dim, kernel, kernel,
                transfer=transfer, w_init=jinit.normal(sigma),
                b_init=jinit.normal(sigma)),
            xnn.MaxPool(pool)))
    # inputs -> [features0, features1, ...]
    dense = xnn.DenseSequential(*layers)
    # [features0, features1, ...] -> [outputs0, outputs1, ...]
    shared = xnn.SharedParallel(xnn.Sequential(
        # features, shape=(f, l) -> features, shape=(l, f)
        xnn.Transpose(),
        # features, shape=(l, f) -> outputs, shape=(l, o)
        transfer(),
        module.ResLinear(
            feat_dim, out_dim, transfer, w_init=jinit.normal(sigma),
            b_init=jinit.normal(sigma)),
        # outputs, shape=(l, o) -> outputs, shape=(o, l)
        xnn.Transpose()))
    return xnn.Sequential(dense, shared)


def Injector(sigma):
    """Noise injector."""
    return xnn.Sequential(
        # inputs -> [inputs] -> [inputs, inputs]
        xnn.Pack(), xnn.Group([0, 0]),
        # [inputs, inputs] -> [inputs, noise]
        xnn.Parallel(
            # inputs -> inputs
            xnn.Identity(),
            # inputs -> noise
            xnn.Sequential(xnn.NormalLike(), xnn.MulConst(sigma))),
        # [inputs, noise] -> inputs + noise
        xnn.Add())


def Random():
    """Random number generator."""
    return xnn.NormalLike()


def AELoss():
    """Auto-Encoder loss. The loss is log-softmax on log-sigmoid, defined as
    Prob(i) = (1 + exp(y_i)) / sum_j(1 + exp(y_j))
    """
    # [outputs, targets] -> loss
    return xnn.Sequential(
        # [outputs, targets] -> [logsoftmax, targets]
        xnn.Parallel(
            xnn.Sequential(
                xnn.MulConst(-1), xnn.LogSigmoid(), xnn.LogSoftmax(axis=0)),
            xnn.Identity()),
        # [logsoftmax, targets] -> loss
        xnn.Multiply(), xnn.Mean())


def GenLoss():
    """Generator loss. LogCosh with zero."""
    # outputs, pytree -> loss, scalar
    return xnn.Sequential(
        # outputs -> [outputs] -> [outputs, outputs]
        xnn.Pack(), xnn.Group([0, 0]),
        # [outputs, outputs] -> [outputs, zeros]
        xnn.Parallel(xnn.Identity(), xnn.ZerosLike()),
        # [outputs, zeros] -> loss
        xnn.LogCosh(), xnn.Mean(), xnn.Concatenate(), xnn.Mean())


def DiscLoss():
    """Discriminator loss. LogCosh with zero for real, negative one for fake."""
    # [real, fake] -> loss
    return nn.Sequential(
        # [real, fake] -> [[real, real], [fake, fake]]
        xnn.Group([[0, 0], [1, 1]]),
        # [[real, real], [fake, fake]] -> [real_loss, fake_loss]
        xnn.Parallel(
            # [real, real] -> [real, zeros] -> real_loss
            xnn.Sequential(
                xnn.Parallel(xnn.Identity(), xnn.ZerosLike()), xnn.LogCosh()),
            # [fake, fake] -> [fake, -ones] -> fake_loss
            xnn.Sequential(
                xnn.Parallel(xnn.Identity(), xnn.FullLike(-1)), xnn.LogCosh())),
        # [real_loss, fake_loss] -> real_loss + fake_loss
        xnn.Add(), xnn.Mean(), xnn.Concatenate(), xnn.Mean())
