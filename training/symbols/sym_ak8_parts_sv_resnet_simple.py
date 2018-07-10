import mxnet as mx

def FC(data, num_hidden, p=None, name=None, suffix=''):
    fc = mx.sym.FullyConnected(data, num_hidden=num_hidden, name='%s%s_fc' % (name, suffix))
    act = mx.sym.Activation(fc, act_type='relu', name='%s%s_relu' % (name, suffix))
    if not p:
        return act
    else:
        dropout = mx.sym.Dropout(act, p=p, name='%s%s_dropout' % (name, suffix))
        return dropout

def residual_unit(data, num_filter, stride, dim_match, name, height=1, bottle_neck=True, bn_mom=0.1, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """

    k_h = min(height, 3)
    pad_h = 0 if height == 1 else 1

    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, k_h), stride=stride, pad=(1, pad_h),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, k_h), stride=stride, pad=(1, pad_h),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, k_h), stride=(1, 1), pad=(1, pad_h),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet(input_name, units, filter_list, num_classes, height=1, bottle_neck=True, bn_mom=0.1, workspace=256, memonger=False):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    k_h = min(height, 3)
    pad_h = 0 if height == 1 else 1

    num_stages = len(units)
    data = mx.sym.Variable(name=input_name)

    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='%s_bn_data' % input_name)

    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, k_h), stride=(1, 1), pad=(1, pad_h),
                          no_bias=True, name="%s_conv0" % input_name, workspace=workspace)

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], stride=(1 if i == 0 else 2, 1 if i == 0 else 2), dim_match=False,
                             height=height, name='%s_stage%d_unit%d' % (input_name, i + 1, 1), bottle_neck=bottle_neck,
                             workspace=workspace, memonger=memonger)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], stride=(1, 1), dim_match=True, height=height, name='%s_stage%d_unit%d' % (input_name, i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='%s_bn1' % input_name)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='%s_relu1' % input_name)
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(1, 1), pool_type='avg', name='%s_pool1' % input_name)
    flat = mx.symbol.Flatten(data=pool1, name='%s_flatten' % input_name)
    return flat

def get_subnet(num_classes, input_name, height=1, filter_list=[64, 128, 256, 512, 1024], bottle_neck=True, units=[3, 4, 6, 3], conv_workspace=256, **kwargs):
    return resnet(units=units,
                  input_name=input_name,
                  filter_list=filter_list,
                  height=height,
                  num_classes=num_classes,
                  bottle_neck=bottle_neck,
                  workspace=conv_workspace,
                  **kwargs)


def get_symbol(num_classes, **kwargs):

    part = get_subnet(num_classes, input_name='part', filter_list=[32, 64, 64, 128], bottle_neck=False, units=[2, 2, 2])
    sv = get_subnet(num_classes, input_name='sv', filter_list=[32, 32, 64], bottle_neck=False, units=[2, 2])

    concat = mx.sym.Concat(*[part, sv], name='concat')

    fc1 = FC(concat, 512, p=0.2, name='fc1')
    fc_out = mx.sym.FullyConnected(fc1, num_hidden=num_classes, name='fc_out')

    softmax = mx.sym.SoftmaxOutput(data=fc_out, name='softmax')

    return softmax

