"""Neural network modules."""

import jax.nn.initializers as jinit
import jax.random as jrand
from xjax import xnn
from xjax.xnn import tree_forward, ModuleTuple
from xjax.xnn import Linear, Conv, Deconv, ResizeLike, ReLU


def Residual(module1, module2, transfer=ReLU, resize=ResizeLike):
    """Residually connected layer with resizing."""
    # inputs -> res_out
    return xnn.Sequence(
        # inputs -> [inputs] -> [inputs, inputs]
        xnn.pack(), xnn.Group([0, 0]),
        # [inputs, inputs] -> [outputs, inputs]
        xnn.Parallel(
            # inputs -> inputs
            xnn.Identity(),
            # inputs -> outputs
            xnn.Sequential(module1, transfer(), module2)),
        # [inputs, outputs] -> [resize(inputs), outputs] -> res_out
        resize(), xnn.Add())


def ResLinear(in_dim, feat_dim, out_dim, transfer=ReLU, resize=ResizeLike,
              w_init=jinit.normal(1e-6), b_winit=jinit.normal(1e-6), rng=None):
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
    deconv2 = Deonv(feat_dim, out_dim, second_kernel, second_stride,
                    second_dilation, second_padding, w_init, b_init, rng2)
    return Residual(deconv1, deconv2, transfer, resize)
