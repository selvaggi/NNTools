import mxnet as mx
from mxnet import gluon
from adversarial.blocks.resnext import ResNeXt
from adversarial.blocks.mdn import SELU, DenseNet

class JNet(gluon.nn.HybridBlock):
    def __init__(self, classes, **kwargs):
        super(JNet, self).__init__(**kwargs)
        with self.name_scope():
            self._pfcand = ResNeXt(bottle_neck=False, layers=[2, 2, 2], channels=[32, 32, 64, 64], prefix='part_')
            self._sv = ResNeXt(bottle_neck=False, layers=[2, 2], channels=[32, 32, 64], prefix='sv_')

            self._fc1 = gluon.nn.Dense(512,
                                       weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
            self._selu = SELU()
#             self._dropout = gluon.nn.Dropout(0.2)
            self._fc_out = gluon.nn.Dense(classes,
                                          weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))

    def hybrid_forward(self, F, pfcand, sv):
        pfcand = self._pfcand(pfcand)
        sv = self._sv(sv)
        concat = F.concat(pfcand, sv)
        fc = self._fc1(concat)
        act = self._selu(fc)
        fc_out = self._fc_out(act)
        log_softmax = F.log_softmax(fc_out)
        return log_softmax


def make_symD(num_classes, **kwargs):

    pfcand = mx.sym.var('part')
    sv = mx.sym.var('sv')

    netD = JNet(num_classes)
    netD.hybridize()

    symD = netD(pfcand, sv)
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
