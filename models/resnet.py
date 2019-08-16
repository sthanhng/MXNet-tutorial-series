"""ResNets, implemented in Gluon."""

from __future__ import division

__all__ = []

from mxnet import nd
from mxnet.gluon import nn


# Blocks
class BasicBlock(nn.HybridBlock):
    r"""BasicBlock from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """
    def __init__(self, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        
    def hybrid_forward(self, F, x):
        pass
    
   
class BottleNeck(nn.HybridBlock):
    r"""Bottleneck from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """
    def __init__(self, **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        
    def hybrid_forward(self, F, X):
        pass
