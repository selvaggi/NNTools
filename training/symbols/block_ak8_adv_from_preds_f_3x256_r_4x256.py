import mxnet as mx
from mxnet import gluon
from adversarial.blocks.mdn import DenseNet

def make_symD(num_classes, **kwargs):

    netD = DenseNet(num_layers=3, hidden_units=256, num_classes=num_classes)
    netD.hybridize()
    symD = netD(mx.sym.var('data'))
    softmax = mx.sym.exp(data=symD, name='softmax')

    return netD, symD, softmax

def get_net(num_classes, use_softmax=False, **kwargs):
    netD, symD, softmax = make_symD(num_classes)
    netD.hybridize()

    netAdv = DenseNet(num_layers=4, hidden_units=256, num_classes=kwargs['adv_mass_nbins'])  # Not converging w/ DropOut???
    netAdv.hybridize()
    symAdv = netAdv(mx.sym.var('scores'))

    if use_softmax:
        return netD, netAdv, symD, symAdv, softmax
    else:
        return netD, netAdv, symD, symAdv

def get_loss(**kwargs):
    lossD = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=True)
    lossD.hybridize()
    lossAdv = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=True)
    lossAdv.hybridize()
    return lossD, lossAdv
