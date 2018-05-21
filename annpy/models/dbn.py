import torch
from torch.nn import Parameter, ModuleList, Module
from .base import LearningRule
from .rbm import RBM

class GreddyOptimizer(object):
    def __init__(self, model, *optimizers):
        if len(optimizers) != model.nr_layers():
            raise Exception("Expected one optimizer per layer")

        if not all(optimizer.model is layer
                   for layer, optimizer in zip(model.layers, optimizers)):
            raise Exception("Optimizer not correpond with model")

        self._model = model
        self.optimizers = optimizers

    @property
    def model(self):
        return self._model

    def step(self, batch):
        for i, learning_rule in enumerate(self.optimizers):
            transformed_batch = self._model.sample_h_given_v(batch, i)
            learning_rule.step(transformed_batch)


class DBN(Module):
    def __init__(self):
        super(DBN, self).__init__()
        self._layers = ModuleList()

    def append(self, *models):
        for model in models:
            if len(self._layers) > 0 and  \
                model.visible_units != self._layers[-1].hidden_units:
                raise Exception("Bad dimentions")

            self._layers.append(model)

    def sample_h_given_v(self, visible, k=None):
        x = visible
        if k is None:
            k = len(self._layers)
        for i in range(0, k):
            module = self._layers[i]
            x = module.sample_h_given_v(x)
        return x

    def sample_v_given_h(self, hidden, k=-1):
        x = hidden
        for i in range(len(self._layers)-1, k, -1):
            module = self._layers[i]
            x = module.sample_v_given_h(x)
        return x

    def reconstruct(self, visible, k=1):
        v_sample = visible
        for _ in range(k):
            h_sample = self.sample_h_given_v(v_sample)
            v_sample = self.sample_v_given_h(h_sample)
        return v_sample

    def nr_layers(self):
        return len(self._layers)

    @property
    def layers(self):
        return self._layers
