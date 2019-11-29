import mxnet as mx
import mxnet.gluon.nn as nn

bn_momentum = 0.9


def get_shape(x):
    if isinstance(x, mx.nd.NDArray):
        return x.shape
    elif isinstance(x, mx.symbol.Symbol):
        _, x_shape, _ = x.infer_shape_partial()
        return x_shape[0]


# A shape is (N, C, P_A), B shape is (N, C, P_B)
# D shape is (N, P_A, P_B)
class BatchDistanceMatrix(nn.HybridBlock):

    def __init__(self):
        super(BatchDistanceMatrix, self).__init__()

    def hybrid_forward(self, F, A, B):
        r_A = F.sum(A * A, axis=1, keepdims=True)  # (N, 1, P_A)
        r_B = F.sum(B * B, axis=1, keepdims=True)  # (N, 1, P_B)
        m = F.batch_dot(F.transpose(A, axes=(0, 2, 1)), B)  # (N, P_A, P_B)
        D = F.broadcast_add(F.broadcast_sub(F.transpose(r_A, axes=(0, 2, 1)), 2 * m), r_B)
        return D


class NearestNeighborsFromIndices(nn.HybridBlock):

    def __init__(self, K, cpu_mode=False):
        super(NearestNeighborsFromIndices, self).__init__()
        self.K = K
        self.cpu_mode = cpu_mode

    def hybrid_forward(self, F, topk_indices, features):
        # topk_indices: (N, P, K)
        # features: (N, C, P)
        queries_shape = get_shape(features)
        batch_size = queries_shape[0]
        channel_num = queries_shape[1]
        point_num = queries_shape[2]

        if self.cpu_mode:
            features = F.transpose(features, (0, 2, 1))  # (N, P, C)
            point_indices = topk_indices  # (N, P, K)
            batch_indices = F.tile(F.reshape(F.arange(batch_size), (-1, 1, 1)), (1, point_num, self.K))  # (N, P, K)
            indices = F.concat(batch_indices.expand_dims(0), point_indices.expand_dims(0), dim=0)  # (2, N, P, K)
            nn_fts = F.gather_nd(features, indices)  # (N, P, K, C)
            return F.transpose(nn_fts, (0, 3, 1, 2))  # (N, C, P, K)
        else:
            point_indices = topk_indices.expand_dims(axis=1).tile((1, channel_num, 1, 1))  # (N, C, P, K)
            batch_indices = F.tile(F.reshape(F.arange(batch_size), (-1, 1, 1, 1)), (1, channel_num, point_num, self.K))  # (N, C, P, K)
            channel_indices = F.tile(F.reshape(F.arange(channel_num), (1, -1, 1, 1)), (batch_size, 1, point_num, self.K))  # (N, C, P, K)
            indices = F.concat(batch_indices.expand_dims(0), channel_indices.expand_dims(0), point_indices.expand_dims(0), dim=0)  # (3, N, C, P, K)
            return F.gather_nd(features, indices)


class Dense(nn.HybridBlock):

    def __init__(self, output, drop_rate=0, activation='relu'):
        super(Dense, self).__init__()
        self.net = nn.Dense(units=output, flatten=False)
        if activation is None:
            self.act = None
        elif activation.lower() == 'selu':
            self.act = nn.SELU()
        else:
            self.act = nn.Activation(activation)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def hybrid_forward(self, F, x):
        x = self.net(x)
        if self.act is not None:
            x = self.act(x)
        if self.drop is not None:
            x = self.drop(x)
        return x


class EdgeConv(nn.HybridBlock):

    def __init__(self, K, channels, in_channels=0, with_bn=True, activation='relu', pooling='average', cpu_mode=False):
        """EdgeConv
        Args:
            K: int, number of neighbors
            channels: tuple of convolution channels
            in_channels: # of input channels
            pooling: 'max', 'average'
        Inputs:
            points: (N, P, C_in)
        Returns:
            transformed points: (N, P, C_out), C_out = channels[-1]
        """
        super(EdgeConv, self).__init__()
        self.K = K
        self.pooling = pooling
        if self.pooling not in ('max', 'average'):
            raise RuntimeError('Pooling method should be "max" or "average"')
        with self.name_scope():
            self.batch_distance_matrix = BatchDistanceMatrix()
            self.knn = NearestNeighborsFromIndices(K, cpu_mode=cpu_mode)
            self.convs = []
            self.bns = []
            self.acts = []
            for idx, C in enumerate(channels):
                self.convs.append(nn.Conv2D(channels=C, kernel_size=(1, 1), strides=(1, 1), use_bias=False if with_bn else True, in_channels=2 * in_channels if idx == 0 else channels[idx - 1], weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)))
                self.register_child(self.convs[-1], 'conv_%d' % idx)
                self.bns.append(nn.BatchNorm(momentum=bn_momentum) if with_bn else None)
                self.register_child(self.bns[-1], 'bn_%d' % idx)
                self.acts.append(nn.Activation(activation) if activation else None)
                self.register_child(self.acts[-1], 'act_%d' % idx)
            if channels[-1] == in_channels:
                self.sc_conv = None
            else:
                self.sc_conv = nn.Conv1D(channels=channels[-1], kernel_size=1, strides=1, use_bias=False if with_bn else True, in_channels=in_channels, weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
                self.sc_bn = nn.BatchNorm(momentum=bn_momentum) if with_bn else None
            self.sc_act = nn.Activation(activation) if activation else None

    def hybrid_forward(self, F, points, features):
        # points: (N, C_p, P)
        # features: (N, C_0, P)

        # distances
        D = self.batch_distance_matrix(points, points)  # (N, P, P)
        indices = F.topk(D, axis=-1, k=self.K + 1, ret_typ='indices', is_ascend=True, dtype='float32')  # (N, P, K+1)
        indices = F.slice_axis(indices, axis=-1, begin=1, end=None)  # (N, P, K)

        fts = features
        knn_fts = self.knn(indices, fts)  # (N, C, P, K)
        knn_fts_center = F.tile(F.expand_dims(fts, axis=3), (1, 1, 1, self.K))  # (N, C, P, K)
        knn_fts = F.concat(knn_fts_center, knn_fts - knn_fts_center, dim=1)  # (N, C, P, K)

        # conv
        x = knn_fts
        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        if self.pooling == 'max':
            fts = F.max(x, axis=-1)  # (N, C, P)
        else:
            fts = F.mean(x, axis=-1)  # (N, C, P)

        # shortcut
        if self.sc_conv:
            sc = self.sc_conv(features)  # (N, C_out, P)
            if self.sc_bn:
                sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.HybridBlock):

    def __init__(self, setting, **kwargs):
        super(ParticleNet, self).__init__(**kwargs)
        self.xconv_params = setting.xconv_params
        self.fc_params = setting.fc_params
        self.num_class = setting.num_class
        self.ec_pooling = getattr(setting, 'pooling', 'average')  # default to average
        if self.ec_pooling not in ('max', 'average'):
            raise RuntimeError('Pooling method should be "max" or "average"')

        with self.name_scope():
            self.bn_fts = nn.BatchNorm(momentum=bn_momentum)
            self.xconvs = nn.HybridSequential()
            for layer_idx, layer_param in enumerate(self.xconv_params):
                K, channels = layer_param
                if layer_idx == 0:
                    in_channels = 0
                else:
                    in_channels = self.xconv_params[layer_idx - 1][1][-1]
                xc = EdgeConv(K, channels, with_bn=True, activation='relu', pooling=self.ec_pooling, in_channels=in_channels, cpu_mode=getattr(setting, 'cpu_mode', False))
                self.xconvs.add(xc)

            if self.fc_params is not None:
                self.fcs = nn.HybridSequential()
                for layer_idx, layer_param in enumerate(self.fc_params):
                    channel_num, drop_rate = layer_param
                    self.fcs.add(Dense(channel_num, drop_rate))
                self.fcs.add(Dense(self.num_class, activation=None))

    def hybrid_forward(self, F, points, features=None, mask=None):

        # points : (N, C_position, P)
        fts = self.bn_fts(features)  # (N, C_features, P)

        # mask: (N, 1, P)
        if mask is not None:
            mask = (mask != 0)  # 1 if valid
            coord_shift = (mask == 0) * 99.  # 99 if non-valid

        for layer_idx, layer_param in enumerate(self.xconv_params):
#             pts = points if layer_idx == 0 else F.broadcast_add(coord_shift, fts)
            pts = F.broadcast_add(coord_shift, points) if layer_idx == 0 else F.broadcast_add(coord_shift, fts)
            fts = self.xconvs[layer_idx](pts, fts)

        if mask is not None:
            fts = F.broadcast_mul(fts, mask)

        pool = F.mean(fts, axis=-1)  # (N, C)

        if self.fc_params is not None:
            logits = self.fcs(pool)  # (N, num_classes)
            return logits
        else:
            return pool


class FeatureConv(nn.HybridBlock):

    def __init__(self, channels, in_channels=0, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv1D(channels, kernel_size=1, strides=1, in_channels=in_channels, use_bias=False, weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)))
        self.body.add(nn.BatchNorm(momentum=bn_momentum))
        self.body.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        x = self.body(x)
        return x
