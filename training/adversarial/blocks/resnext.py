import mxnet as mx
from mxnet.gluon import nn

bn_momentum = 0.3

# Helpers
def _conv3xh(channels, height, stride, groups=1):
    pad_h = 0 if height == 1 else 1
    return nn.Conv2D(channels, kernel_size=(3, height), strides=stride, padding=(1, pad_h),
                     use_bias=False, groups=groups,
                     weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))


class BasicBlock(nn.HybridBlock):
    r"""BasicBlock V3 from
    `"Aggregated Residual Transformations for Deep Neural Networks"
    <https://arxiv.org/abs/1611.05431>`_ paper.
    This is used for ResNet V3 for 18, 34 layers.
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
    def __init__(self, channels, height, stride, downsample=False, groups=1, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = _conv3xh(channels, height, stride)
            self.bn1 = nn.BatchNorm(momentum=bn_momentum)
            self.act1 = nn.Activation('relu')
            self.conv2 = _conv3xh(channels, height, 1)
            self.bn2 = nn.BatchNorm(momentum=bn_momentum)
            self.act2 = nn.Activation('relu')
            if downsample:
                self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                            weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
                self.bn_sc = nn.BatchNorm(momentum=bn_momentum)
            else:
                self.downsample = None

    def hybrid_forward(self, F, x):
        shortcut = x
        if self.downsample:
            shortcut = self.downsample(x)
            shortcut = self.bn_sc(shortcut)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.act2(x + shortcut)


class Bottleneck(nn.HybridBlock):
    r"""Bottleneck V3 from
    `"Aggregated Residual Transformations for Deep Neural Networks"
    <https://arxiv.org/abs/1611.05431>`_ paper.
    This is used for ResNet V3 for 50, 101, 152 layers.
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
    def __init__(self, channels, height, stride, downsample=False, groups=32, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels // 2, kernel_size=1, strides=1, use_bias=False,
                                   weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
            self.bn1 = nn.BatchNorm(momentum=bn_momentum)
            self.act1 = nn.Activation('relu')
            self.conv2 = _conv3xh(channels // 2, height, stride, groups=groups)
            self.bn2 = nn.BatchNorm(momentum=bn_momentum)
            self.act2 = nn.Activation('relu')
            self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False,
                                   weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
            self.bn3 = nn.BatchNorm(momentum=bn_momentum)
            self.act3 = nn.Activation('relu')
            if downsample:
                self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                            weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
                self.bn_sc = nn.BatchNorm(momentum=bn_momentum)
            else:
                self.downsample = None

    def hybrid_forward(self, F, x):
        shortcut = x
        if self.downsample:
            shortcut = self.downsample(x)
            shortcut = self.bn_sc(shortcut)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        return self.act3(x + shortcut)


class ResNeXt(nn.HybridBlock):
    r"""BasicBlock V3 from
    `"Aggregated Residual Transformations for Deep Neural Networks"
    <https://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int
        Number of classification classes.
    height : int, default 1
        Height of the input.
    """
    def __init__(self, bottle_neck, layers, channels, height=1, groups=32, **kwargs):
        super(ResNeXt, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        self.height = height
        self.groups = groups
        block = Bottleneck if bottle_neck else BasicBlock
        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(nn.BatchNorm(scale=False, center=False, momentum=bn_momentum))
            if height == 1:
                self.features.add(_conv3xh(channels[0], height=height, stride=1))
            else:
                self.features.add(nn.Conv2D(channels[0], kernel_size=7, strides=2, padding=3, use_bias=False,
                                            weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)))
                self.features.add(nn.BatchNorm(momentum=bn_momentum))
                self.features.add(nn.Activation('relu'))
#                 self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i + 1],
                                                   stride, i + 1))

            self.classifier = nn.HybridSequential()
            self.classifier.add(nn.GlobalAvgPool2D())
            self.classifier.add(nn.Flatten())

    def _make_layer(self, block, layers, channels, stride, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, height=self.height, stride=stride, downsample=True,
                            groups=self.groups))
            for _ in range(layers - 1):
                layer.add(block(channels, height=self.height, stride=1, downsample=False,
                                groups=self.groups))
        return layer

    def hybrid_forward(self, F, x):
        if self.height != 1:
            x = F.log1p(100 * x)
        x = self.features(x)
        x = self.classifier(x)
        return x
