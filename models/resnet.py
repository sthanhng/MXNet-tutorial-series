"""ResNets, implemented in Gluon."""

from __future__ import division

__all__ = []

from mxnet import nd
from mxnet.gluon import nn


# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                    use_bias=False, in_channels=in_channels)


# Blocks
class BasicBlock(nn.HybridBlock):
    r"""BasicBlock from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)

        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))

        self.body.add(_conv3x3(channels, 1, in_channels))
        self.body.add(nn.BatchNorm())

        if downsample:
            self.shortcut = nn.HybridSequential(prefix='')
            self.shortcut.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                        use_bias=False, in_channels=in_channels))
        else:
            self.shortcut = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.shortcut:
            residual = self.shortcut(residual)
        
        x = F.Activation(residual + x, act_type='relu')

        return x
    
   
class BottleNeck(nn.HybridBlock):
    r"""Bottleneck from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleNeck, self).__init__(**kwargs)

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels // 4, kernel_size=1, strides=stride))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))

        self.body.add(_conv3x3(channels // 4, 1, channels // 4))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))

        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())

        if downsample:
            self.shortcut = nn.HybridSequential(prefix='')
            self.shortcut.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                        use_bias=False, in_channels=in_channels))
        else:
            self.shortcut = None
        
    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.shortcut:
            residual = self.shortcut(residual)
        
        x = F.Activation(residual + x, act_type='relu')

        return x
