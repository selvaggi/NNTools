import mxnet as mx

def FC(data, num_hidden, p=None, name=None, suffix=''):
    fc = mx.sym.FullyConnected(data, num_hidden=num_hidden, name='%s%s_fc' % (name, suffix))
    act = mx.sym.Activation(fc, act_type='relu', name='%s%s_relu' % (name, suffix))
    if not p:
        return act
    else:
        dropout = mx.sym.Dropout(act, p=p, name='%s%s_dropout' % (name, suffix))
        return dropout

def residual_unit(data, num_filter, stride, dim_match, name, height=1, bottle_neck=True, num_group=32, bn_mom=0.1, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
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
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter * 0.5), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')


        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.5), num_group=num_group, kernel=(3, k_h), stride=stride, pad=(1, pad_h),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')


        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name + '_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')

        eltwise = bn3 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3, k_h), stride=stride, pad=(1, pad_h),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')


        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, k_h), stride=(1, 1), pad=(1, pad_h),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name + '_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')

        eltwise = bn2 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')

def resnext(input_name, units, filter_list, height=1, bottle_neck=True, num_group=32, bn_mom=0.1, workspace=256, memonger=False):
    """Return ResNeXt symbol of
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

#     body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, k_h), stride=(1, 1), pad=(1, pad_h),
#                           no_bias=True, name="%s_conv0" % input_name, workspace=workspace)

    if height == 1:
        data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='%s_bn_data' % input_name)
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, k_h), stride=(1, 1), pad=(1, 0),
                              no_bias=True, name="%s_conv0" % input_name, workspace=workspace)
    else:
        data = mx.sym.log1p(100 * data, name='%s_log1p' % input_name)  # !!!

        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                              no_bias=True, name="%s_conv0" % input_name, workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
#        body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], stride=(1 if i == 0 else 2, 1 if i == 0 else 2), dim_match=False,
                             height=height, name='%s_stage%d_unit%d' % (input_name, i + 1, 1), bottle_neck=bottle_neck, num_group=num_group,
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], stride=(1, 1), dim_match=True, height=height, name='%s_stage%d_unit%d' % (input_name, i + 1, j + 2),
                                 bottle_neck=bottle_neck, num_group=num_group, bn_mom=bn_mom, workspace=workspace, memonger=memonger)

    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=body, global_pool=True, kernel=(1, 1), pool_type='avg', name='%s_pool1' % input_name)
    flat = mx.symbol.Flatten(data=pool1, name='%s_flatten' % input_name)
    return flat

def get_subnet(input_name, height=1, filter_list=[64, 128, 256, 512, 1024], bottle_neck=True, units=[3, 4, 6, 3], num_group=32, conv_workspace=256, **kwargs):
    return resnext(units=units,
                  input_name=input_name,
                  filter_list=filter_list,
                  height=height,
                  bottle_neck=bottle_neck,
                  num_group=num_group,
                  workspace=conv_workspace,
                  **kwargs)

def get_symbol(num_classes, **kwargs):

    pfcand = get_subnet(input_name='img', height=3, filter_list=[64, 64, 128, 256, 512], bottle_neck=True, units=[3, 4, 6, 3])
    pf_dropout = mx.sym.Dropout(pfcand, p=0.5, name='pf_dropout')
    fc_out = mx.sym.FullyConnected(pf_dropout, num_hidden=num_classes, name='fc_out')
    softmax = mx.sym.SoftmaxOutput(data=fc_out, name='softmax')

    return softmax

