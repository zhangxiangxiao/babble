"""Neural network modules."""

import jax.nn.initializers as jinit
import jax.random as jrand
from xjax.xnn import *


def Residual(module1, module2, transfer=Tanh, resize=ResizeLike):
    """Residually connected layer with resizing."""
    # inputs -> res_out
    return Sequential(
        # inputs -> [inputs] -> [inputs, inputs]
        Pack(), Group([0, 0]),
        # [inputs, inputs] -> [outputs, inputs]
        Parallel(Identity(), Sequential(module1, transfer(), module2)),
        # [inputs, outputs] -> [resize(inputs), outputs] -> res_out
        resize(), Add())


def ResLinear(in_dim, feat_dim, out_dim, transfer=Tanh, resize=ResizeLike,
              w_init=jinit.normal(1e-6), b_init=jinit.normal(1e-6), rng=None):
    """Residually connected linear layer."""
    rng1, rng2 = jrand.split(rng) if rng is not None else (None, None)
    linear1 = Linear(in_dim, feat_dim, w_init, b_init, rng1)
    linear2 = Linear(feat_dim, out_dim, w_init, b_init, rng2)
    return Residual(linear1, linear2, transfer, resize)


def ResConv(in_dim, feat_dim, out_dim, first_kernel, second_kernel,
            first_stride=None, second_stride=None, first_dilation=None,
            second_dilation=None, first_padding='SAME', second_padding='SAME',
            transfer=Tanh, resize=ResizeLike, w_init=jinit.normal(1e-6),
            b_init=jinit.normal(1e-6), rng=None):
    """Residually connected convolution."""
    rng1, rng2 = jrand.split(rng) if rng is not None else (None, None)
    conv1 = Conv(in_dim, feat_dim, first_kernel, first_stride, first_dilation,
                 first_padding, w_init, b_init, rng1)
    conv2 = Conv(feat_dim, out_dim, second_kernel, second_stride,
                 second_dilation, second_padding, w_init, b_init, rng2)
    return Residual(conv1, conv2, transfer, resize)


def ResDeconv(in_dim, feat_dim, out_dim, first_kernel, second_kernel,
              first_stride=None, second_stride=None, first_dilation=None,
              second_dilation=None, first_padding='SAME', second_padding='SAME',
              transfer=Tanh, resize=ResizeLike, w_init=jinit.normal(1e-6),
              b_init=jinit.normal(1e-6), rng=None):
    """Residually connected deconvolution."""
    rng1, rng2 = jrand.split(rng) if rng is not None else (None, None)
    deconv1 = Deconv(in_dim, feat_dim, first_kernel, first_stride,
                     first_dilation, first_padding, w_init, b_init, rng1)
    deconv2 = Deconv(feat_dim, out_dim, second_kernel, second_stride,
                     second_dilation, second_padding, w_init, b_init, rng2)
    return Residual(deconv1, deconv2, transfer, resize)


def Encoder(level, depth, in_dim, feat_dim, out_dim, kernel=(3,), pool=(2,),
            sigma=1e-6, transfer=Tanh):
    """Encoder."""
    layers = []
    # inputs -> features
    layers.append(ResConv(
        in_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
        w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    for _ in range(depth - 1):
        # features -> features
        layers.append(transfer())
        layers.append(ResConv(
            feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
            w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    # features -> features.
    for _ in range(level - 1):
        # features, shape=(f, l) -> features, shape=(f, l/2)
        layers.append(MaxPool(pool))
        for _ in range(depth):
            # features -> features
            layers.append(transfer())
            layers.append(ResConv(
                feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
                w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    # features, shape=(f, l) -> features, shape=(f, l/2)
    layers.append(MaxPool(pool))
    for _ in range(depth - 1):
        # features -> features
        layers.append(transfer())
        layers.append(ResConv(
            feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
            w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    # features -> outputs
    layers.append(transfer())
    layers.append(ResConv(
        feat_dim, feat_dim, out_dim, kernel, kernel, transfer=transfer,
        w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    # outputs -> outputs
    layers.append(Standardize(axis=0))
    return Sequential(*layers)


def Decoder(level, depth, in_dim, feat_dim, out_dim, kernel=(3,), stride=(2,),
            sigma=1e-6, transfer=Tanh):
    """Decoder."""
    layers = []
    # inputs -> inputs
    layers.append(Standardize(axis=0))
    # inputs -> features
    layers.append(ResConv(
        in_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
        w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    for _ in range(depth - 1):
        # features -> features
        layers.append(transfer())
        layers.append(ResConv(
            feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
            w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    # features -> features
    for _ in range(level - 1):
        # features, shape=(f, l) -> features, shape=(f, l*2)
        layers.append(transfer())
        layers.append(ResDeconv(
            feat_dim, feat_dim, feat_dim, kernel, kernel, stride,
            transfer=transfer, w_init=jinit.normal(sigma),
            b_init=jinit.normal(sigma)))
        for _ in range(depth - 1):
            # features -> features
            layers.append(transfer())
            layers.append(ResConv(
                feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
                w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    # features, shape=(f, l) -> features, shape=(f, l*2)
    if depth > 1:
        layers.append(transfer())
        layers.append(ResDeconv(
            feat_dim, feat_dim, feat_dim, kernel, kernel, stride,
            transfer=transfer, w_init=jinit.normal(sigma),
            b_init=jinit.normal(sigma)))
        for _ in range(depth - 2):
            # features -> features
            layers.append(transfer())
            layers.append(ResConv(
                feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
                w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
        # features -> outputs
        layers.append(transfer())
        layers.append(ResConv(
            feat_dim, feat_dim, out_dim, kernel, kernel, transfer=transfer,
            w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    else:
        layers.append(transfer())
        layers.append(ResDeconv(
            feat_dim, feat_dim, out_dim, kernel, kernel, stride,
            transfer=transfer, w_init=jinit.normal(sigma),
            b_init=jinit.normal(sigma)))
    return Sequential(*layers)


def Discriminator(level, depth, in_dim, feat_dim, out_dim, kernel=(3,),
                  pool=(2,), dropout=0.5, sigma=1e-6, transfer=Tanh):
    """Discriminator that is dense."""
    layers = []
    layers.append(Sequential(
        # inputs -> features
        ResConv(in_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
                w_init=jinit.normal(sigma), b_init=jinit.normal(sigma))))
    for _ in range(depth - 1):
        # features -> features
        layers.append(Sequential(
            Dropout(dropout), transfer(),
            ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
    # features -> features.
    for _ in range(level - 1):
        # features, shape=(f, l) -> features, shape=(f, l/2)
        layers.append(Sequential(
            Dropout(dropout), MaxPool(pool), transfer(),
            ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
        for _ in range(depth - 1):
            # features -> features
            layers.append(Sequential(
                Dropout(dropout), transfer(),
                ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                        transfer=transfer, w_init=jinit.normal(sigma),
                        b_init=jinit.normal(sigma))))
    if depth > 1:
        # features, shape=(f, l) -> features, shape=(f, l/2)
        layers.append(Sequential(
            Dropout(dropout), MaxPool(pool), transfer(),
            ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
        for _ in range(depth - 2):
            # features -> features
            layers.append(Sequential(
                Dropout(dropout), transfer(),
                ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                        transfer=transfer, w_init=jinit.normal(sigma),
                        b_init=jinit.normal(sigma))))
        # features -> outputs
        layers.append(Sequential(
            Dropout(dropout), transfer(),
            ResConv(feat_dim, feat_dim, out_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
    else:
        # features, shape=(f, l) -> features, shape=(f, l/2)
        layers.append(Sequential(
            Dropout(dropout), MaxPool(pool), transfer(),
            ResConv(feat_dim, feat_dim, out_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
    # inputs -> [feature0, feature1, ..., outputs]
    return DenseSequential(*layers)


def FeatureInjector(beta=0.1):
    """Noise injector."""
    return Sequential(
        # inputs -> [inputs] -> [inputs, inputs]
        Pack(), Group([0, 0]),
        # [inputs, inputs] -> [inputs, noise]
        Parallel(Identity(), Sequential(NormalLike(), MulConst(beta))),
        # [inputs, noise] -> inputs + noise
        Add())


def InputInjector(in_dim, beta=0.1):
    """Noise injector to discrete inputs."""
    return Sequential(
        # inputs -> [inputs] -> [inputs, inputs, inputs]
        Pack(), Group([0, 0, 0]),
        # [inputs, inputs] -> [inputs, noise, factor]
        Parallel(
            # inputs -> inputs
            Identity(),
            # inputs -> noise
            Sequential(Mean(axis=0), RandintLike(None, minval=0, maxval=in_dim),
                       OneHot(num_classes=in_dim, axis=0)),
            # inputs -> factor
            Sequential(Mean(axis=0), BernoulliLike(None, p=beta))),
        # [inputs, factor, noise] -> [[inputs, factor], [noise, factor]]
        Group([[0, 2], [1, 2]]),
        # [[inputs, factor], [noise, factor] -> [inputs, noise]
        Parallel(
            # [inputs, factor] -> inputs
            Sequential(
                # [inputs, factor] -> [inputs, 1 - factor]
                Parallel(Identity(), Sequential(MulConst(-1), AddConst(1))),
                # [inputs, 1 - factor] -> inputs
                Multiply()),
            # [noise, factor] -> noise
            Multiply()),
        # [inputs, noise] -> inputs + noise
        Add())


def FeatureRandom():
    """Random number generator."""
    return NormalLike()


def InputRandom(in_dim):
    """Random input generator."""
    return Sequential(Mean(axis=0), RandintLike(None, minval=0, maxval=in_dim),
                      OneHot(num_classes=in_dim, axis=0))


def AELoss(weight=1):
    """Auto-Encoder loss."""
    # [outputs, targets, weights] -> loss
    return Sequential(
        # [outputs, targets, weights] -> [[outputs, targets], weights]
        Group([[0, 1], 2]),
        # [[outputs, targets], weights] -> [loss, weights]
        Parallel(
            # [outputs, targets] -> loss
            Sequential(
                # [outputs, targets] -> [[outputs, targets], [outputs, targets]]
                Group([[0, 1], [0, 1]]),
                # [[outputs, targets], [outputs, targets]] -> [pos_loss, neg_loss]
                Parallel(
                    # [outputs, targets] -> pos_loss
                    Sequential(
                        # [outputs, targets] -> [logsig, targets]
                        Parallel(Sequential(LogSigmoid(), MulConst(-1)), Identity()),
                        # [logsig, targets] -> [[logsig, targets], targets]
                        Group([[0, 1], 1]),
                        # [[logsig, targets], targets] -> [loss, tar_sum]
                        Parallel(Sequential(Multiply(), Sum(axis=0)), Sum(axis=0)),
                        # [loss, tar_sum] -> loss
                        Divide()),
                    # [outputs, targets -> neg_loss
                    Sequential(
                        # [outputs, targets] -> [logsig, targets]
                        Parallel(Softplus(), Sequential(MulConst(-1), AddConst(1))),
                        # [logsig, targets] -> [[logsig, targets], targets]
                        Group([[0, 1], 1]),
                        # [[logsig, targets], targets] -> [loss, tar_sum]
                        Parallel(Sequential(Multiply(), Sum(axis=0)), Sum(axis=0)),
                        # [loss, tar_sum] -> loss
                        Divide())),
                # [pos_loss, neg_loss] -> loss
                Add()),
            # weights -> weights
            Identity()),
        # [loss, weights] -> loss
        Multiply(), Mean(), MulConst(weight))


def GenLoss(weight=1):
    """Generator loss."""
    # [real, fake] -> loss
    return Sequential(
        # [real, fake] -> [[real, real], [fake, fake]]
        Group([[0, 0], [1, 1]]),
        # [[real, real], [fake, fake]] -> [real_loss, fake_loss]
        Parallel(
            # [real, real] -> [real, zeros] -> real_loss
            Sequential(Parallel(Identity(), ZerosLike()), Subtract(), LogCosh()),
            # [fake, fake] -> [fake, zeros] -> fake_loss
            Sequential(Parallel(Identity(), ZerosLike()), Subtract(), LogCosh())),
        # [real_loss, fake_loss] -> real_loss + fake_loss
        Add(), Mean(), Stack(), Mean(), MulConst(weight))


def DiscLoss(weight=1):
    """Discriminator loss."""
    # [real, fake] -> loss
    return Sequential(
        # [real, fake] -> [[real, real], [fake, fake]]
        Group([[0, 0], [1, 1]]),
        # [[real, real], [fake, fake]] -> [real_loss, fake_loss]
        Parallel(
            # [real, real] -> [real, ones] -> real_loss
            Sequential(Parallel(Identity(), OnesLike()), Subtract(), LogCosh()),
            # [fake, fake] -> [fake, -ones] -> fake_loss
            Sequential(Parallel(Identity(), FullLike(-1)), Subtract(), LogCosh())),
        # [real_loss, fake_loss] -> real_loss + fake_loss
        Add(), Mean(), Stack(), Mean(), MulConst(weight))


def DiscLossSigmoid(weight=1):
    """Discriminator loss. LogCosh for real, logSigmoid for fake."""
    return Sequential(
        # [real, fake] -> [real_loss, fake_loss]
        Parallel(Sequential(MulConst(-1), Softplus()), Softplus()),
        # [real_loss, fake_loss] -> real_loss + fake_loss
        Add(), Mean(), Stack(), Mean(), MulConst(weight))
