# See LICENSE file for copyright and license details.
import torch
from torch.nn import Module, Parameter, Sigmoid, Sequential
from torch.nn.modules import Linear
from torch.autograd import Variable

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
            layers.append(activations[i]())

        self._layers = Sequential(*layers)

    def forward(self, input):
        return self._layers(input)

    def __repr__(self):
        return "{} ({})".format(self.__class__.__name__,
                                " -> ".join(map(repr, self._layers)))
