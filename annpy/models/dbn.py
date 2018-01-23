import torch
from torch.nn import Parameter, ModuleList, Module
from .base import LearningRule
from .rbm import RBM

class GreddyLearningRule(LearningRule):
    def __init__(self, model, *learning_rules):
        if len(learning_rules) != model.nr_layers():
            raise Exception("Expected one learning rule per layer")

        if not all(learning_rule.model is layer
                   for layer, learning_rule in zip(model.layers, learning_rules)):
            raise Exception("Learning rule not correpond with model")

        super(GreddyLearningRule, self).__init__(model)
        self.learning_rules = learning_rules

    def step(self, batch):
        for i, learning_rule in enumerate(self.learning_rules):
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

    def reconstruct_error_metric(self, metric, k):
        return lambda x: metric(self.reconstruct(x, k), x)
