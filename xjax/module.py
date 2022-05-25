"""Neural network modules."""

import jax.nn.initializers as jinit
import jax.random as jrand
from xjax import xnn
from xjax.xnn import tree_forward, ModuleTuple
from xjax.xnn import Linear, Conv, Deconv, ResizeLike, Tanh, Dropout


def Residual(module1, module2, transfer=Tanh, resize=ResizeLike):
    """Residually connected layer with resizing."""
    # inputs -> res_out
    return xnn.Sequential(
        # inputs -> [inputs] -> [inputs, inputs]
        xnn.Pack(), xnn.Group([0, 0]),
        # [inputs, inputs] -> [outputs, inputs]
        xnn.Parallel(
            # inputs -> inputs
            xnn.Identity(),
            # inputs -> outputs
            xnn.Sequential(module1, transfer(), module2)),
        # [inputs, outputs] -> [resize(inputs), outputs] -> res_out
        resize(), xnn.Add())


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
        layers.append(xnn.MaxPool(pool))
        for _ in range(depth):
            # features -> features
            layers.append(transfer())
            layers.append(ResConv(
                feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
                w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    # features, shape=(f, l) -> features, shape=(f, l/2)
    layers.append(xnn.MaxPool(pool))
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
    layers.append(xnn.Standardize(axis=0))
    return xnn.Sequential(*layers)


def Decoder(level, depth, in_dim, feat_dim, out_dim, kernel=(3,), stride=(2,),
            sigma=1e-6, transfer=Tanh):
    """Decoder."""
    layers = []
    # inputs -> inputs
    layers.append(xnn.Standardize(axis=0))
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
    return xnn.Sequential(*layers)


def Discriminator(level, depth, in_dim, feat_dim, out_dim, kernel=(3,),
                  pool=(2,), dropout=0.5, sigma=1e-6, transfer=Tanh):
    """Discriminator that is dense."""
    layers = []
    layers.append(xnn.Sequential(
        # inputs -> features
        xnn.Softplus(), xnn.Softmax(axis=0),
        ResConv(in_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
                w_init=jinit.normal(sigma), b_init=jinit.normal(sigma))))
    for _ in range(depth - 1):
        # features -> features
        layers.append(xnn.Sequential(
            Dropout(dropout),
            transfer(),
            ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
    # features -> features.
    for _ in range(level - 1):
        # features, shape=(f, l) -> features, shape=(f, l/2)
        layers.append(xnn.Sequential(
            Dropout(dropout),
            xnn.MaxPool(pool),
            transfer(),
            ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
        for _ in range(depth - 1):
            # features -> features
            layers.append(xnn.Sequential(
                Dropout(dropout),
                transfer(),
                ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                        transfer=transfer, w_init=jinit.normal(sigma),
                        b_init=jinit.normal(sigma))))
    if depth > 1:
        # features, shape=(f, l) -> features, shape=(f, l/2)
        layers.append(xnn.Sequential(
            Dropout(dropout),
            xnn.MaxPool(pool),
            transfer(),
            ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
        for _ in range(depth - 2):
            # features -> features
            layers.append(xnn.Sequential(
                Dropout(dropout),
                transfer(),
                ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                        transfer=transfer, w_init=jinit.normal(sigma),
                        b_init=jinit.normal(sigma))))
        # features -> outputs
        layers.append(xnn.Sequential(
            Dropout(dropout),
            transfer(),
            ResConv(feat_dim, feat_dim, out_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
    else:
        # features, shape=(f, l) -> features, shape=(f, l/2)
        layers.append(xnn.Sequential(
            Dropout(dropout),
            xnn.MaxPool(pool),
            transfer(),
            ResConv(feat_dim, feat_dim, out_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
    # inputs -> [feature0, feature1, ..., outputs]
    return xnn.DenseSequential(*layers)


def FeatureInjector(beta=0.1):
    """Noise injector."""
    return xnn.Sequential(
        # inputs -> [inputs] -> [inputs, inputs]
        xnn.Pack(), xnn.Group([0, 0]),
        # [inputs, inputs] -> [inputs, noise]
        xnn.Parallel(
            # inputs -> inputs
            xnn.Identity(),
            # inputs -> noise
            xnn.Sequential(xnn.NormalLike(), xnn.MulConst(beta))),
        # [inputs, noise] -> inputs + noise
        xnn.Add())


def InputInjector(in_dim, beta=0.1):
    """Noise injector to discrete inputs."""
    return xnn.Sequential(
        # inputs -> [inputs] -> [inputs, inputs, inputs]
        xnn.Pack(), xnn.Group([0, 0, 0]),
        # [inputs, inputs] -> [inputs, noise, factor]
        xnn.Parallel(
            # inputs -> inputs
            xnn.Identity(),
            # inputs -> noise
            xnn.Sequential(xnn.Mean(axis=0),
                           xnn.RandintLike(None, minval=0, maxval=in_dim),
                           xnn.OneHot(num_classes=in_dim, axis=0)),
            # inputs -> factor
            xnn.Sequential(xnn.Mean(axis=0), xnn.BernoulliLike(None, p=beta))),
        # [inputs, factor, noise] -> [[inputs, factor], [noise, factor]]
        xnn.Group([[0, 2], [1, 2]]),
        # [[inputs, factor], [noise, factor] -> [inputs, noise]
        xnn.Parallel(
            # [inputs, factor] -> inputs
            xnn.Sequential(
                # [inputs, factor] -> [inputs, 1 - factor]
                xnn.Parallel(xnn.Identity(),
                             xnn.Sequential(xnn.MulConst(-1), xnn.AddConst(1))),
                # [inputs, 1 - factor] -> inputs
                xnn.Multiply()),
            # [noise, factor] -> noise
            xnn.Multiply()),
        # [inputs, noise] -> inputs + noise
        xnn.Add())


def FeatureRandom():
    """Random number generator."""
    return xnn.NormalLike()


def InputRandom(in_dim):
    """Random input generator."""
    return xnn.Sequential(
        xnn.Mean(axis=0), xnn.RandintLike(None, minval=0, maxval=in_dim),
        xnn.OneHot(num_classes=in_dim, axis=0))


def AELoss(weight=1):
    """Auto-Encoder loss. The loss is log-softmax on softplus, defined as
    Prob(i) = (1 + exp(y_i)) / sum_j(1 + exp(y_j))
    """
    # [outputs, targets, weights] -> loss
    return xnn.Sequential(
        # [outputs, targets, weights] -> [neglogsoftmax, targets, weights]
        xnn.Parallel(
            xnn.Sequential(
                xnn.Softplus(), xnn.LogSoftmax(axis=0), xnn.MulConst(-1)),
            xnn.Identity(), xnn.Identity()),
        # [neglogsoftmax, targets, weights] -> [loss, weights]
        xnn.Group([[0, 1], 2]), xnn.Parallel(xnn.Multiply(), xnn.Identity()),
        # [loss, weights] -> [[loss, weights], weights] -> [loss, weights]
        xnn.Group([[0, 1], 1]), xnn.Parallel(xnn.Multiply(), xnn.Identity()),
        # [loss, weights] -> [loss_sum, weights_sum] -> loss_mean
        xnn.Parallel(xnn.Sum(), xnn.Sum()), xnn.Divide(), xnn.MulConst(weight))


def GenLoss(weight=1):
    """Generator loss."""
    # [real, fake] -> loss
    return xnn.Sequential(
        # [real, fake] -> [[real, real], [fake, fake]]
        xnn.Group([[0, 0], [1, 1]]),
        # [[real, real], [fake, fake]] -> [real_loss, fake_loss]
        xnn.Parallel(
            # [real, real] -> [real, zeros] -> real_loss
            xnn.Sequential(
                xnn.Parallel(xnn.Identity(), xnn.ZerosLike()), xnn.Subtract(),
                xnn.LogCosh()),
            # [fake, fake] -> [fake, zeros] -> fake_loss
            xnn.Sequential(
                xnn.Parallel(xnn.Identity(), xnn.ZerosLike()), xnn.Subtract(),
                xnn.LogCosh())),
        # [real_loss, fake_loss] -> real_loss + fake_loss
        xnn.Add(), xnn.Reshape(-1), xnn.Concatenate(), xnn.Mean(),
        xnn.MulConst(weight))


def DiscLoss(weight=1):
    """Discriminator loss."""
    # [real, fake] -> loss
    return xnn.Sequential(
        # [real, fake] -> [[real, real], [fake, fake]]
        xnn.Group([[0, 0], [1, 1]]),
        # [[real, real], [fake, fake]] -> [real_loss, fake_loss]
        xnn.Parallel(
            # [real, real] -> [real, ones] -> real_loss
            xnn.Sequential(
                xnn.Parallel(xnn.Identity(), xnn.OnesLike()), xnn.Subtract(),
                xnn.LogCosh()),
            # [fake, fake] -> [fake, -ones] -> fake_loss
            xnn.Sequential(
                xnn.Parallel(xnn.Identity(), xnn.FullLike(-1)), xnn.Subtract(),
                xnn.LogCosh())),
        # [real_loss, fake_loss] -> real_loss + fake_loss
        xnn.Add(), xnn.Reshape(-1), xnn.Concatenate(), xnn.Mean(),
        xnn.MulConst(weight))


def DiscLossSigmoid(weight=1):
    """Discriminator loss. LogCosh for real, logSigmoid for fake."""
    return xnn.Sequential(
        # [real, fake] -> [real_loss, fake_loss]
        xnn.Parallel(
            # real -> real_loss
            xnn.Sequential(xnn.MulConst(-1), xnn.Softplus()),
            # fake -> fake_loss
            xnn.Softplus()),
        # [real_loss, fake_loss] -> real_loss + fake_loss
        xnn.Add(), xnn.Reshape(-1), xnn.Concatenate(), xnn.Mean(),
        xnn.MulConst(weight))
