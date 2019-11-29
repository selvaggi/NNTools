import mxnet as mx
from adversarial.blocks.particle_net import ParticleNet, Dense


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
    output = pf_net(pf_points, pf_features, pf_mask)
    # -------
    softmax = mx.sym.SoftmaxOutput(data=output, name='softmax')

    return softmax
