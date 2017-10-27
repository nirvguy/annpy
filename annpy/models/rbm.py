# See LICENSE file for copyright and license details.
import torch
from torch.nn import Module, Parameter, Sigmoid
from .base import LearningRule

class CDRule(LearningRule):
    """ Online Contrastive Divergence Learning Rule"""
    def parameters_for(self, pattern):
        v_sample, h_sample, v_sample_2, h_sample_2 = self.model.gibbs_sample(pattern)

        positive_gradient = torch.ger(v_sample, h_sample)
        negative_gradient = torch.ger(v_sample_2, h_sample_2)

        return (self._lr * (positive_gradient - negative_gradient),
                self._lr * (v_sample - v_sample_2),
                self._lr * (h_sample - h_sample_2))

    def step(self, batch):
        for index in range(0, len(batch), self.mini_batch_size):
            delta_w = torch.zeros(self.model.weights.data.shape)
            delta_v_biases = torch.zeros(self.model.visible_biases.data.shape)
            delta_h_biases = torch.zeros(self.model.hidden_biases.data.shape)

            n = 0
            for pattern in batch[index:index+self.mini_batch_size]:
                pattern_delta_w, pattern_delta_v_b, pattern_delta_h_b = self.parameters_for(pattern)
                delta_w.add_(pattern_delta_w)
                delta_v_biases.add_(pattern_delta_v_b)
                delta_h_biases.add_(pattern_delta_h_b)
                n += 1

            self.model.weights.data.add_(delta_w.div_(n))
            self.model.visible_biases.data.add_(delta_v_biases.div_(n))
            self.model.hidden_biases.data.add_(delta_h_biases.div_(n))

    @property
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value):
        self._lr = torch.Tensor([value]).type(self.dtype)

    def __init__(self, model, lr=0.01, mini_batch_size=1, dtype=torch.FloatTensor):
        super(CDRule, self).__init__(model)
        self.dtype = dtype
        self.learning_rate = lr
        self.mini_batch_size = mini_batch_size

class RBM(Module):
    """ Restricted Boltzman Machine model """

    visible_activation_module = Sigmoid
    hidden_activation_module = Sigmoid

    visible_sampler = torch.bernoulli
    hidden_sampler = torch.bernoulli
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

        self._visible_activation = self.__class__.visible_activation_module()
        self._hidden_activation = self.__class__.hidden_activation_module()

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

    def _net_visible(self, hidden):
        """ Return the net visible stimulus given a hidden sample vector
        """
        x=hidden.matmul(self.weights.data.t())

        return x + self.visible_biases.data

    def _net_hidden(self, visible):
        """ Return the net hidden stimulus given a visible pattern vector
        """
        x=visible.matmul(self.weights.data)

        return x + self.hidden_biases.data

    def reconstruct(self, visible, k=1):
        """ Reconstruct the true visible pattern

        Args:
            visible: Visble input pattern
            k (int): Number of iterations
        """
        for _ in range(k):
            visible = self.sample_v_given_h(self.sample_h_given_v(visible))
        return visible

    def proba_h_given_v(self, visible):
        """ Probability vector of hidden state given visible pattern vector
        """
        return self._hidden_activation(self._net_hidden(visible))

    def proba_v_given_h(self, hidden):
        """ Probability vector of visible pattern given hidden state vector
        """
        return self._visible_activation(self._net_visible(hidden))

    def sample_h_given_v(self, visible):
        """ Sample a hidden state given a visible pattern
        """
        return self.__class__.hidden_sampler(self.proba_h_given_v(visible))

    def sample_v_given_h(self, hidden):
        """ Sample a visible pattern given hidden state vector
        """
        return self.__class__.visible_sampler(self.proba_v_given_h(hidden))

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
        h_sample_2 = torch.Tensor(h_sample)

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
