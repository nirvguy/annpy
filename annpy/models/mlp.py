# See LICENSE file for copyright and license details.
import torch
from torch.nn import Module, Parameter, Sigmoid, Sequential
from torch.nn.modules import Linear
from torch.nn import MSELoss
from torch.autograd import Variable
from torch.optim import SGD
from .base import LearningRule
from .activations import Softmax

class MLP(Module):
    def __init__(self,
                 units,
                 activations=None):
        super(MLP, self).__init__()
        layers = []

        if activations is None:
            activations = [Sigmoid] * (len(units)-1)

        for i in range(len(units)-1):
            layers.append(Linear(units[i], units[i+1]))
            layers.append(activations[i])

        self._layers = Sequential(*layers)

    def forward(self, input):
        return self._layers(input)

    def __repr__(self):
        return "{} ({})".format(self.__class__.__name__,
                                " -> ".join(map(repr, self._layers)))

class MLPClassifier(MLP):
    def __init__(self, units, classes, activations=None):
        total_units = units + [len(classes)]

        if activations is None:
            activations = [Sigmoid] * (len(units)-1)
        activations.append(Softmax)

        super(MLPClassifier, self).__init__(total_units, activations)
        self._classes = classes

    def proba(x):
        return self.forward(x)

    def predict(x):
        proba = self.proba(x)

        if x.dim() == 1:
            _, pos_max = proba.max(0)
        else:
            _, pos_max = proba.max(1)

        return self._classes[pos_max.data]
