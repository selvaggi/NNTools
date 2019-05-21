import mxnet as mx
from mxnet import gluon
import numpy as np

class SELU(gluon.nn.Activation):
    """Applies selu activation function to input."""
    def __init__(self, **kwargs):
        self._act_type = 'selu'
        super(SELU, self).__init__('selu', **kwargs)

    def _alias(self):
        return self._act_type

    def hybrid_forward(self, F, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * F.LeakyReLU(x, act_type='elu', slope=alpha)

    def __repr__(self):
        s = '{name}({_act_type})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


class DenseNet(gluon.nn.HybridBlock):
    def __init__(self, num_layers, hidden_units, num_classes=None, apply_log_softmax=True, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.apply_log_softmax = apply_log_softmax
        with self.name_scope():
            # first build a body as feature
            self.body = gluon.nn.HybridSequential()
            for i in range(num_layers):
                self.body.add(gluon.nn.Dense(hidden_units, weight_initializer=mx.init.Xavier()))
                self.body.add(SELU())
            if num_classes is not None:
                self.body.add(gluon.nn.Dense(num_classes, weight_initializer=mx.init.Xavier()))

    def hybrid_forward(self, F, x):
        fc_out = self.body(x)
        if self.num_classes is not None and self.apply_log_softmax:
            return F.log_softmax(fc_out)
        else:
            return fc_out

class DenseNetDropout(gluon.nn.HybridBlock):
    def __init__(self, num_layers, hidden_units, num_classes=None, drop=None, apply_log_softmax=True, **kwargs):
        super(DenseNetDropout, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.apply_log_softmax = apply_log_softmax
        with self.name_scope():
            # first build a body as feature
            self.body = gluon.nn.HybridSequential()
            for i in range(num_layers):
                self.body.add(gluon.nn.Dense(hidden_units[i], weight_initializer=mx.init.Xavier()))
                self.body.add(gluon.nn.LeakyReLU(0.2))
                if drop is not None and drop[i] is not None:
                    self.body.add(gluon.nn.Dropout(drop[i]))
            if num_classes is not None:
                self.body.add(gluon.nn.Dense(num_classes, weight_initializer=mx.init.Xavier()))

    def hybrid_forward(self, F, x):
        fc_out = self.body(x)
        if self.num_classes is not None and self.apply_log_softmax:
            return F.log_softmax(fc_out)
        else:
            return fc_out

class MixtureDensityNetwork(gluon.nn.HybridBlock):
    def __init__(self, num_layers, hidden_units,
                 num_components=10, num_output_features=1, epsilon=1.e-3, **kwargs):
        super(MixtureDensityNetwork, self).__init__(**kwargs)
        self.num_components = num_components  # number of gaus components
        self.num_output_features = num_output_features  # dim of output features
        self.epsilon = epsilon
        with self.name_scope():
            # first build a body as feature
            self.body = gluon.nn.HybridSequential()
            for i in range(num_layers):
                self.body.add(gluon.nn.Dense(hidden_units, weight_initializer=mx.init.Xavier()))
                self.body.add(SELU())
            # mean: linear activation
            self.mu = gluon.nn.Dense(num_components * num_output_features, weight_initializer=mx.init.Xavier())
            # sigma: must be positive [exp activcation, to be applied]
            self.sigma = gluon.nn.Dense(num_components, weight_initializer=mx.init.Xavier())
            # coef alpha: must be (0,1) [softmax activation, to be applied]
            self.alpha = gluon.nn.Dense(num_components, weight_initializer=mx.init.Xavier())

    def hybrid_forward(self, F, x):
        body = self.body(x)
        mu = self.mu(body)
        sigma = F.LeakyReLU(self.sigma(body), act_type='elu', slope=1) + 1 + self.epsilon
        alpha = F.clip(F.softmax(self.alpha(body)), self.epsilon, 1)
        return mu, sigma, alpha

class NLLLoss(gluon.nn.HybridBlock):
    def __init__(self, num_components, num_features=1, epsilon=1.e-3, weight=None, batch_axis=0, **kwargs):
        super(NLLLoss, self).__init__(**kwargs)
        self._weight = weight
        self._batch_axis = batch_axis
        self._num_components = num_components
        self._num_features = num_features
        self._epsilon = epsilon

    def log_sum_exp(self, F, x, axis=None):
        """Log-sum-exp trick implementation"""
        x_max = F.max(x, axis=axis, keepdims=True)
        return F.log(self._epsilon + F.sum(F.exp(F.broadcast_minus(x , x_max)), axis=axis, keepdims=True)) + x_max

    def hybrid_forward(self, F, mu, sigma, alpha, label, sample_weight=None):
        ''' data shapes:
             - mu: NM[C] (batch, component, feature)
             - sigma: NM
             - alpha: NM
             - label: N[C]
        '''
        mu = mu.reshape((-1, self._num_components, self._num_features))
        label = label.reshape((-1, 1, self._num_features))

        exponent = F.broadcast_minus(
            F.broadcast_minus(
                F.log(alpha) - float(.5 * self._num_features * np.log(2 * np.pi)),
                float(self._num_features) * F.log(sigma)
                ),
            0.5 * F.sum(F.square(F.broadcast_minus(label, mu)), axis=2) / F.square(sigma)
        )

        nll = -self.log_sum_exp(F, exponent, axis=1)
        nll = gluon.loss._apply_weighting(F, nll, self._weight, sample_weight)
        return F.mean(nll, axis=self._batch_axis, exclude=True)
