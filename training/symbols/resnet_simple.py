import mxnet as mx
from symbols.resnet_base import get_subnet, FC

def get_symbol(num_classes, **kwargs):

# v2
#     pfcand = get_subnet(num_classes, input_name='pfcand', filter_list=[32, 32, 32, 64, 64], bottle_neck=False, units=[3, 4, 6, 3])
#     track = get_subnet(num_classes, input_name='track', filter_list=[32, 32, 64, 64, 128], bottle_neck=False, units=[3, 4, 6, 3])
#     sv = get_subnet(num_classes, input_name='sv', filter_list=[32, 32, 64, 64], bottle_neck=False, units=[2, 2, 2])

# v2.1 simple
    pfcand = get_subnet(num_classes, input_name='pfcand', filter_list=[32, 32, 64, 64], bottle_neck=False, units=[2, 2, 2])
    track = get_subnet(num_classes, input_name='track', filter_list=[32, 64, 64, 128], bottle_neck=False, units=[2, 2, 2])
    sv = get_subnet(num_classes, input_name='sv', filter_list=[32, 32, 64], bottle_neck=False, units=[2, 2])

    concat = mx.sym.Concat(*[pfcand, track, sv], name='concat')

    fc1 = FC(concat, 512, p=0.2, name='fc1')
    fc_out = mx.sym.FullyConnected(fc1, num_hidden=num_classes, name='fc_out')

    softmax = mx.sym.SoftmaxOutput(data=fc_out, name='softmax')

    return softmax

