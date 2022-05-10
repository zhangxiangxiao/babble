"""Neural network modules."""

import jax.nn.initializers as jinit
import jax.random as jrand
from xjax import xnn
from xjax.xnn import tree_forward, ModuleTuple
from xjax.xnn import Linear, Conv, Deconv, ResizeLike, ReLU


def Residual(module1, module2, transfer=ReLU, resize=ResizeLike):
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


def ResLinear(in_dim, feat_dim, out_dim, transfer=ReLU, resize=ResizeLike,
              w_init=jinit.normal(1e-6), b_init=jinit.normal(1e-6), rng=None):
    """Residually connected linear layer."""
    rng1, rng2 = jrand.split(rng) if rng is not None else (None, None)
    linear1 = Linear(in_dim, feat_dim, w_init, b_init, rng1)
    linear2 = Linear(feat_dim, out_dim, w_init, b_init, rng2)
    return Residual(linear1, linear2, transfer, resize)


def ResConv(in_dim, feat_dim, out_dim, first_kernel, second_kernel,
            first_stride=None, second_stride=None, first_dilation=None,
            second_dilation=None, first_padding='SAME', second_padding='SAME',
            transfer=ReLU, resize=ResizeLike, w_init=jinit.normal(1e-6),
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
              transfer=ReLU, resize=ResizeLike, w_init=jinit.normal(1e-6),
              b_init=jinit.normal(1e-6), rng=None):
    """Residually connected deconvolution."""
    rng1, rng2 = jrand.split(rng) if rng is not None else (None, None)
    deconv1 = Deconv(in_dim, feat_dim, first_kernel, first_stride,
                     first_dilation, first_padding, w_init, b_init, rng1)
    deconv2 = Deconv(feat_dim, out_dim, second_kernel, second_stride,
                     second_dilation, second_padding, w_init, b_init, rng2)
    return Residual(deconv1, deconv2, transfer, resize)


def Encoder(level, depth, in_dim, feat_dim, out_dim, kernel=(3,), pool=(2,),
            sigma=1e-6, transfer=xnn.ReLU):
    """Encoder."""
    layers = []
    # inputs, shape=(i, l) -> inputs, shape=(l, i)
    layers.append(xnn.Transpose())
    # inputs -> features
    layers.append(ResLinear(
        in_dim, feat_dim, feat_dim, transfer, w_init=jinit.normal(sigma),
        b_init=jinit.normal(sigma)))
    # features, shape=(l, f) -> features, shape=(f, l)
    layers.append(xnn.Transpose())
    for _ in range(depth):
        # features -> features
        layers.append(transfer())
        layers.append(ResConv(
            feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
            w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    # features -> features.
    for _ in range(level):
        # features, shape=(f, l) -> features, shape=(f, l/2)
        layers.append(xnn.MaxPool(pool))
        for _ in range(depth):
            # features -> features
            layers.append(transfer())
            layers.append(ResConv(
                feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
                w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    # features, shape=(f, l) -> features, shape=(l, f)
    layers.append(xnn.Transpose())
    # features, shape=(l, f) -> outputs, shape=(l, o)
    layers.append(transfer())
    layers.append(ResLinear(
        feat_dim, feat_dim, out_dim, transfer, w_init=jinit.normal(sigma),
        b_init=jinit.normal(sigma)))
    # outputs, shape=(l, o) -> outputs, shape=(o, l)
    layers.append(xnn.Transpose())
    # features -> features
    layers.append(xnn.Standardize(axis=0))
    return xnn.Sequential(*layers)


def Decoder(level, depth, in_dim, feat_dim, out_dim, kernel=(3,), stride=(2,),
            sigma=1e-6, transfer=xnn.ReLU):
    """Decoder."""
    layers = []
    # inputs -> inputs
    layers.append(xnn.Standardize(axis=0))
    # inputs, shape=(i, l) -> inputs, shape=(l, i)
    layers.append(xnn.Transpose())
    # inputs -> features
    layers.append(ResLinear(
        in_dim, feat_dim, feat_dim, transfer, w_init=jinit.normal(sigma),
        b_init=jinit.normal(sigma)))
    # features, shape=(l, f) -> features, shape=(f, l)
    layers.append(xnn.Transpose())
    for _ in range(depth):
        # features -> features
        layers.append(transfer())
        layers.append(ResConv(
            feat_dim, feat_dim, feat_dim, kernel, kernel, transfer=transfer,
            w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)))
    # features -> features
    for _ in range(level):
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
    # features, shape=(f, l) -> features, shape=(l, f)
    layers.append(xnn.Transpose())
    # features, shape=(l, f) -> outputs, shape=(l, o)
    layers.append(transfer())
    layers.append(ResLinear(
        feat_dim, feat_dim, out_dim, transfer, w_init=jinit.normal(sigma),
        b_init=jinit.normal(sigma)))
    # outputs, shape=(l, o) -> outputs, shape=(o, l)
    layers.append(xnn.Transpose())
    return xnn.Sequential(*layers)


def Discriminator(level, depth, in_dim, feat_dim, out_dim, kernel=(3,),
                  pool=(2,), sigma=1e-6, transfer=xnn.ReLU):
    """Discriminator that is dense."""
    layers = []
    layers.append(xnn.Sequential(
        # inputs -> inputs, as softmax on softplus.
        xnn.Softplus(), xnn.Softmax(axis=0),
        # inputs, shape=(i, l) -> inputs, shape=(l, i)
        xnn.Transpose(),
        # inputs -> features
        ResLinear(in_dim, feat_dim, feat_dim, transfer,
                  w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)),
        # features, shape=(l, f) -> features, shape=(f, l)
        xnn.Transpose(),
        transfer(),
        ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                transfer=transfer, w_init=jinit.normal(sigma),
                b_init=jinit.normal(sigma))))
    for _ in range(depth - 1):
        # features -> features
        layers.append(xnn.Sequential(
            transfer(),
            ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
    # features -> features.
    for _ in range(level):
        # features, shape=(f, l) -> features, shape=(f, l/2)
        layers.append(xnn.Sequential(
            xnn.MaxPool(pool),
            transfer(),
            ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                    transfer=transfer, w_init=jinit.normal(sigma),
                    b_init=jinit.normal(sigma))))
        for _ in range(depth - 1):
            # features -> features
            layers.append(xnn.Sequential(
                transfer(),
                ResConv(feat_dim, feat_dim, feat_dim, kernel, kernel,
                        transfer=transfer, w_init=jinit.normal(sigma),
                        b_init=jinit.normal(sigma))))
    # inputs -> [features0, features1, ...]
    dense = xnn.DenseSequential(*layers)
    # [features0, features1, ...] -> [outputs0, outputs1, ...]
    shared = xnn.SharedParallel(xnn.Sequential(
        # features, shape=(f, l) -> features, shape=(l, f)
        xnn.Transpose(),
        # features, shape=(l, f) -> outputs, shape=(l, o)
        transfer(),
        ResLinear(feat_dim, feat_dim, out_dim, transfer,
                  w_init=jinit.normal(sigma), b_init=jinit.normal(sigma)),
        # outputs, shape=(l, o) -> outputs, shape=(o, l)
        xnn.Transpose()))
    return xnn.Sequential(dense, shared)


def FeatureInjector(beta=1):
    """Noise injector."""
    return xnn.Sequential(
        # inputs -> [inputs] -> [inputs, [inputs, inputs]]
        xnn.Pack(), xnn.Group([0, [0, 0]]),
        # [inputs, [inputs, inputs]] -> [inputs, noise]
        xnn.Parallel(
            # inputs -> inputs
            xnn.Identity(),
            # [inputs, inputs] -> noise
            xnn.Sequential(
                # [inputs, inputs] -> [noise, scale]
                xnn.Parallel(
                    xnn.NormalLike(),
                    xnn.Sequential(xnn.Exponential(), xnn.MulConst(beta))),
                # [noise, scale] -> noise
                xnn.Multiply())),
        # [inputs, noise] -> inputs + noise
        xnn.Add())


def FeatureRandom():
    """Random number generator."""
    return xnn.NormalLike()


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
    """Generator loss. LogCosh with zero."""
    # outputs, pytree -> loss, scalar
    return xnn.Sequential(
        # outputs -> [outputs] -> [outputs, outputs]
        xnn.Pack(), xnn.Group([0, 0]),
        # [outputs, outputs] -> [outputs, zeros]
        xnn.Parallel(xnn.Identity(), xnn.ZerosLike()),
        # [outputs, zeros] -> loss
        xnn.LogCosh(), xnn.Mean(), xnn.Stack(), xnn.Mean(),
        xnn.MulConst(weight))


def DiscLoss(weight=1):
    """Discriminator loss. LogCosh with zero for real, negative one for fake."""
    # [real, fake] -> loss
    return xnn.Sequential(
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
        xnn.Add(), xnn.Mean(), xnn.Stack(), xnn.Mean(), xnn.MulConst(weight))


def DiscLossSigmoid(weight=1):
    """Discriminator loss. LogCosh for real, logSigmoid for fake."""
    return xnn.Sequential(
        # [real, fake] -> [[real, real], fake]
        xnn.Group([[0, 0], 1]),
        # [[real, real], fake] -> [real_loss, fake_loss]
        xnn.Parallel(
            # [real, real] -> [real, zeros] -> real_loss
            xnn.Sequential(xnn.Parallel(
                xnn.Identity(), xnn.ZerosLike()), xnn.LogCosh()),
            # fake -> fake_loss
            xnn.SoftPlus()),
        # [real_loss, fake_loss] -> real_loss + fake_loss
        xnn.Add(), xnn.Mean(), xnn.Stack(), xnn.Mean(), xnn.MulConst(weight))
