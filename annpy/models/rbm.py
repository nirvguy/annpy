# See LICENSE file for copyright and license details.
import torch
from torch.autograd import Variable
from torch.nn import Module, Parameter, Sigmoid
from .base import LearningRule

class CDOptimizer(object):
    """ Contrastive Divergence Optimizer """
    def update(self, pattern):
        v_sample, h_sample, v_sample_2, h_sample_2 = self._model.gibbs_sample(pattern)

        positive_gradient = torch.ger(v_sample, h_sample)
        negative_gradient = torch.ger(v_sample_2, h_sample_2)

        self.deltas[self._model.weights].add_(self._lr * (positive_gradient - negative_gradient).data)
        self.deltas[self._model.visible_biases].add_(self._lr * (v_sample - v_sample_2).data)
        self.deltas[self._model.hidden_biases].add_(self._lr * (h_sample - h_sample_2).data)

    def reset_deltas(self):
        self.deltas[self._model.weights] = torch.zeros(self._model.weights.data.shape)
        self.deltas[self._model.visible_biases] = torch.zeros(self._model.visible_biases.data.shape)
        self.deltas[self._model.hidden_biases] = torch.zeros(self._model.hidden_biases.data.shape)

    def step(self, batch):
        self.deltas = {}

        self.reset_deltas()

        for pattern in batch:
            self.update(pattern)

        for parameter, update in self.deltas.items():
            parameter.data.add_(update / len(batch))

        del self.deltas

    @property
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value):
        self._lr = float(value)

    def __init__(self, model, lr=0.01, dtype=torch.FloatTensor):
        self._model = model
        self.dtype = dtype
        self.learning_rate = lr

class RBM(Module):
    """ Restricted Boltzman Machine model """
    @staticmethod
    def random_uniform_tensor(dtype, shape, std=0.01):
        return torch.normal(means=torch.zeros(*shape), std=std).type(dtype)

    def __init__(self,
                 nr_visible, nr_hiddens,
                 weights=None, visible_biases=None, hidden_biases=None,
                 dtype=torch.FloatTensor):
        """ Model constructor

          Args:
              nr_visible (int):              Number of visible units
              nr_hidden  (int)               Number of hidden units
              weights (torch.Tensor):        Initial weights. None to setup it randomly (default)
              visible_biases (torch.Tensor): Initial biases for visible biases unit's. None to setupit randomly (default)
              hidden_biases (torch.Tensor):  Initial biases for hidden biases unit's. None to setupit randomly (default)
        """
        super(RBM, self).__init__()
        self._nr_visible = nr_visible
        self._nr_hiddens = nr_hiddens

        self.reset_parameters(weights, visible_biases, hidden_biases, dtype=dtype)

    def reset_parameters(self,
                         initial_weights,
                         initial_visible_biases,
                         initial_hidden_biases,
                         dtype=torch.FloatTensor):
        """ Reset the network parameters such as weights and hidden and visible biases
        """

        if initial_weights is not None:
            self.weights = Parameter(initial_weights)
        else:
            self.weights = Parameter(self.random_uniform_tensor(dtype, (self._nr_visible, self._nr_hiddens)))

        if initial_visible_biases is not None:
            self.visible_biases = Parameter(initial_visible_biases)
        else:
            self.visible_biases = Parameter(self.random_uniform_tensor(dtype, (self._nr_visible, )))

        if initial_hidden_biases is not None:
            self.hidden_biases = Parameter(initial_hidden_biases)
        else:
            self.hidden_biases = Parameter(self.random_uniform_tensor(dtype, (self._nr_hiddens, )))

        self.visible_activation = Sigmoid()
        self.hidden_activation = Sigmoid()

    def _net_visible(self, hidden):
        """ Return the net visible stimulus given a hidden sample vector
        """
        x=hidden.matmul(self.weights.t())

        return x + self.visible_biases

    def _net_hidden(self, visible):
        """ Return the net hidden stimulus given a visible pattern vector
        """
        x=visible.matmul(self.weights)

        return x + self.hidden_biases

    def reconstruct(self, visible, k=1):
        """ Reconstruct the true visible pattern

        Args:
            visible: Visble input pattern
            k (int): Number of iterations
        """
        for _ in range(k):
            visible = self.sample_v_given_h(self.sample_h_given_v(visible))
        return visible

    def sample_h_given_v(self, visible):
        """ Sample a hidden state given a visible pattern
        """
        probabilities = self.hidden_activation(self._net_hidden(visible))
        return torch.bernoulli(probabilities)

    def sample_v_given_h(self, hidden):
        """ Sample a visible pattern given hidden state vector
        """
        probabilities = self.visible_activation(self._net_visible(hidden))
        return torch.bernoulli(probabilities)

    @property
    def visible_units(self):
        return self._nr_visible

    @property
    def hidden_units(self):
        return self._nr_hiddens

    def gibbs_sample(self, pattern, samples=1):
        """ Performs a gibbs sampling

          Args:
            pattern: Visible pattern
            samples: Number of sampling gibbs iterations

          Returns:
            A tuple (pattern, h_sample, v_sample, h_sample_2) were
            h_sample is a hidden state sample generated from `pattern`,
            v_sample is a visible pattern generated from `h_sample` after samples iterations
            and h_sample_2 is a hidden state sample generated from `v_sample`
        """
        h_sample = self.sample_h_given_v(pattern)

        v_sample = None
        h_sample_2 = h_sample

        for _ in range(samples):
            v_sample = self.sample_v_given_h(h_sample_2)
            h_sample_2 = self.sample_h_given_v(v_sample)

        return pattern, h_sample, v_sample, h_sample_2

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self._nr_visible) + ' -> ' \
               + str(self._nr_hiddens) + ')'

    forward = sample_h_given_v

    def reconstruct_error_metric(self, metric):
        return lambda x: metric(self.reconstruct(x), x)
