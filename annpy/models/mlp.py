# See LICENSE file for copyright and license details.
import torch
from torch.nn import Module, Parameter, Sigmoid, Sequential
from torch.nn.modules import Linear
from torch.nn import MSELoss
from torch.autograd import Variable
from torch.optim import SGD
from .base import LearningRule

class Backpropagation(LearningRule):
    def __init__(self, 
                 model,
                 loss_fn=MSELoss(),
                 learning_rate=0.001,
                 mini_batch_size=1,
                 optimizer_class=SGD):
        super(Backpropagation, self).__init__(model)
        self.loss_fn = loss_fn
        self.mini_batch_size = mini_batch_size
        self._optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    @property
    def mini_batch_size(self):
        return self._mini_batch_size

    @mini_batch_size.setter
    def mini_batch_size(self, mini_batch_size):
        if not isinstance(mini_batch_size, int):
            raise Exception("Expected an integer mini_batch_size but got {}".format(type(mini_batch_size)))

        if mini_batch_size < 1:
            raise Exception("Expected mini_batch_size greater than zero but got {}".format(mini_batch_size))

        self._mini_batch_size = mini_batch_size

    def step(self, batch_input, batch_output):
        for index in range(0, len(batch_input), self._mini_batch_size):
            x = Variable(batch_input[index:index+self._mini_batch_size])
            y = Variable(batch_output[index:index+self._mini_batch_size])
            self._optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self._optimizer.step()

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
