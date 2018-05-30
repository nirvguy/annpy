# See LICENSE file for copyright and license details.
import torch
from torch.nn import Module, Parameter, Sigmoid, Sequential, BatchNorm1d, Dropout
from torch.nn.modules import Linear
from torch.nn import MSELoss
from torch.autograd import Variable
from torch.optim import SGD
from .base import LearningRule
from .activations import Softmax

class MLP(Module):
    def __init__(self,
                 units,
                 activations=None,
                 batch_norm=False,
                 dropouts=None):
        super(MLP, self).__init__()
        layers = []

        if activations is None:
            activations = [Sigmoid()] * (len(units)-1)

        if dropouts is None:
            dropouts = [None] * (len(units)-1)

        for i in range(len(units)-1):
            layers.append(Linear(units[i], units[i+1]))
            if activations[i] is not None:
                layers.append(activations[i])
            if batch_norm:
                layers.append(BatchNorm1d(units[i+1]))
            if dropouts[i] is not None:
                layers.append(Dropout(p=dropouts[i]))

        self._layers = Sequential(*layers)

    def forward(self, input):
        return self._layers(input)

    def __repr__(self):
        return "{} ({})".format(self.__class__.__name__,
                                " -> ".join(map(repr, self._layers)))

class MLPClassifier(MLP):
    def __init__(self, units, classes, activations=None, batch_norm=False, dropouts=None):
        total_units = units + [len(classes)]

        if activations is None:
            activations = [Sigmoid()] * (len(units)-1)
        activations.append(None)

        super(MLPClassifier, self).__init__(total_units, activations, batch_norm=batch_norm, dropouts=dropouts)
        self._classes = classes
        self.softmax = Softmax()

    def proba(self, batch):
        return self.softmax(self.forward(batch))

    def predict(self, batch):
        proba = self.proba(batch)

        pos_max, _ = proba.topk(k=1)
        pos_max = pos_max.data

        return [self._classes[p] for p in pos_max]
