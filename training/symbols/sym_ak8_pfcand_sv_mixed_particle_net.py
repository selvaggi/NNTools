import mxnet as mx
from adversarial.blocks.particle_net import ParticleNet, FeatureConv


class DotDict:
    pass


def split_batch_size(shape, k):
    return (shape[0] // k,) + shape[1:]


def get_symbol(num_classes, **kwargs):
    # pfcand
    pf_setting = DotDict()
    # K, C
    pf_setting.xconv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    pf_setting.fc_params = [(256, 0.1)]
    pf_setting.num_class = num_classes
    pf_setting.pooling = 'average'
    pf_setting.cpu_mode = (kwargs['gpus'] == '')
    pf_setting.n_split = max(1, len(kwargs['gpus'].split(',')))

    pf_net = ParticleNet(pf_setting, prefix="ParticleNet_pfcand_")
    pf_net.hybridize()

    pf_points = mx.sym.var('pf_points', shape=split_batch_size(kwargs['data_shapes']['pf_points'], pf_setting.n_split))
    pf_features = mx.sym.var('pf_features', shape=split_batch_size(kwargs['data_shapes']['pf_features'], pf_setting.n_split))
    pf_mask = mx.sym.var('pf_mask', shape=split_batch_size(kwargs['data_shapes']['pf_mask'], pf_setting.n_split))

    pf_fts_conv = FeatureConv(channels=32, in_channels=kwargs['data_shapes']['pf_features'][1])
    pf_fts_conv.hybridize()
    pf_features = pf_fts_conv(pf_features)

    sv_points = mx.sym.var('sv_points', shape=split_batch_size(kwargs['data_shapes']['sv_points'], pf_setting.n_split))
    sv_features = mx.sym.var('sv_features', shape=split_batch_size(kwargs['data_shapes']['sv_features'], pf_setting.n_split))
    sv_mask = mx.sym.var('sv_mask', shape=split_batch_size(kwargs['data_shapes']['sv_mask'], pf_setting.n_split))

    sv_fts_conv = FeatureConv(channels=32, in_channels=kwargs['data_shapes']['sv_features'][1])
    sv_fts_conv.hybridize()
    sv_features = sv_fts_conv(sv_features)

    points = mx.sym.concat(pf_points, sv_points, dim=-1)
    features = mx.sym.concat(pf_features, sv_features, dim=-1)
    mask = mx.sym.concat(pf_mask, sv_mask, dim=-1)

    output = pf_net(points, features, mask)
    # -------
    softmax = mx.sym.SoftmaxOutput(data=output, name='softmax')

    return softmax
