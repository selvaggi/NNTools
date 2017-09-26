import mxnet as mx

def FC(data, num_hidden, p=None, name=None, suffix=''):
    fc = mx.sym.FullyConnected(data, num_hidden=num_hidden, name='%s%s_fc' % (name, suffix))
    act = mx.sym.Activation(fc, act_type='relu', name='%s%s_relu' % (name, suffix))
    if not p:
        return act
    else:
        dropout = mx.sym.Dropout(act, p=p, name='%s%s_dropout' % (name, suffix))
        return dropout

def get_symbol(num_classes, bn_mom=0.1, **kwargs):

    data = mx.sym.Variable(name='img')
#     data = mx.sym.Pooling(data=data, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type='sum')  # downsample to 0.05*0.05
#     data = mx.sym.Pooling(data=data, kernel=(4, 4), stride=(4, 4), pad=(0, 0), pool_type='sum')  # downsample to 0.1*0.1

    body = mx.sym.Convolution(data=data, num_filter=32, kernel=(11, 11), stride=(1, 1), pad=(5, 5),
                      no_bias=True, name="conv0")
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
#     body = mx.sym.Dropout(data=body, p=0.2, name='dropout0')
    body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    body = mx.sym.Convolution(data=body, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      no_bias=True, name="conv1")
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu1')
#     body = mx.sym.Dropout(data=body, p=0.2, name='dropout1')
    body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool1')

    body = mx.sym.Convolution(data=body, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      no_bias=True, name="conv2")
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn2')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu2')
#     body = mx.sym.Dropout(data=body, p=0.2, name='dropout2')
    body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool2')

    body = mx.sym.flatten(data=body, name='flat')
    body = mx.sym.Dropout(data=body, p=0.2, name='flat_dropout')
    body = FC(data=body, num_hidden=64, p=0.1, name='fc0')

    fc_out = mx.sym.FullyConnected(body, num_hidden=num_classes, name='fc_out')
    softmax = mx.sym.SoftmaxOutput(data=fc_out, name='softmax')

    return softmax
