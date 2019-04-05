import mxnet as mx
from mxnet import gluon
from adversarial.blocks.resnext import ResNeXt
from adversarial.blocks.mdn import DenseNetDropout


class JNet(gluon.nn.HybridBlock):
    def __init__(self, classes, **kwargs):
        super(JNet, self).__init__(**kwargs)
        with self.name_scope():
            self._pfcand = ResNeXt(bottle_neck=False, layers=[2, 2, 2], channels=[32, 64, 64, 128], prefix='pfcand_')
            self._sv = ResNeXt(bottle_neck=False, layers=[2, 2], channels=[32, 32, 64], prefix='sv_')
            self._fcs = DenseNetDropout(num_layers=3, hidden_units=[256, 256, 256], drop=[0.2, 0.2, 0.2], num_classes=classes)

    def hybrid_forward(self, F, pfcand, sv):
        pfcand = self._pfcand(pfcand)
        sv = self._sv(sv)
        concat = F.concat(pfcand, sv)
        log_softmax = self._fcs(concat)
        return concat, log_softmax


def make_symD(num_classes, **kwargs):

    pfcand = mx.sym.var('pfcand')
    sv = mx.sym.var('sv')

    netD = JNet(num_classes)
    netD.hybridize()

    _, symD = netD(pfcand, sv)
    softmax = mx.sym.exp(data=symD, name='softmax')

    return netD, symD, softmax

def get_net(num_classes, use_softmax=False, **kwargs):
    netD, symD, softmax = make_symD(num_classes)
    netD.hybridize()

    netAdv = DenseNetDropout(num_layers=3, hidden_units=[256, 256, 256], drop=[0.2, 0.2, 0.2], num_classes=kwargs['adv_mass_nbins'])
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
