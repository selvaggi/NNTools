import mxnet as mx
from mxnet import gluon
from adversarial.blocks.resnext import ResNeXt
from adversarial.blocks.mdn import SELU, DenseNet

class JNet(gluon.nn.HybridBlock):
    def __init__(self, classes, **kwargs):
        super(JNet, self).__init__(**kwargs)
        with self.name_scope():
            self._pfcand = ResNeXt(bottle_neck=False, layers=[2, 2, 2], channels=[32, 64, 64, 128], prefix='part_')
            self._sv = ResNeXt(bottle_neck=False, layers=[2, 2], channels=[32, 32, 64], prefix='sv_')

            self._fc1 = gluon.nn.Dense(512, weight_initializer=mx.init.Xavier())
            self._fc2 = gluon.nn.Dense(256, weight_initializer=mx.init.Xavier())
            self._fc3 = gluon.nn.Dense(256, weight_initializer=mx.init.Xavier())
            self._selu = SELU()

            self._fc_out = gluon.nn.Dense(classes, weight_initializer=mx.init.Xavier())

    def hybrid_forward(self, F, pfcand, sv):
        pfcand = self._pfcand(pfcand)
        sv = self._sv(sv)
        concat = F.concat(pfcand, sv)
        fc1 = self._selu(self._fc1(concat))
        fc2 = self._selu(self._fc2(fc1))
        fc3 = self._selu(self._fc3(fc2))
        fc_out = self._fc_out(fc3)
        log_softmax = F.log_softmax(fc_out)
        return concat, log_softmax


def make_symD(num_classes, **kwargs):

    pfcand = mx.sym.var('part')
    sv = mx.sym.var('sv')

    netD = JNet(num_classes)
    netD.hybridize()

    _, symD = netD(pfcand, sv)
    softmax = mx.sym.exp(data=symD, name='softmax')

    return netD, symD, softmax

def get_net(num_classes, use_softmax=False, **kwargs):
    netD, symD, softmax = make_symD(num_classes)
    netD.hybridize()

    netAdv = DenseNet(num_layers=3, hidden_units=256, num_classes=kwargs['adv_mass_nbins'])  # Not converging w/ DropOut???
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
