import torch
from torch.nn import Module, Parameter, Sigmoid

class RBM(Module):
    visible_activation_module = Sigmoid
    hidden_activation_module = Sigmoid

    visible_sampler = torch.bernoulli
    hidden_sampler = torch.bernoulli

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
        def random_uniform_tensor(*shape):
            return torch.normal(means=torch.zeros(*shape), std=0.01).type(dtype)

        self._visible_activation = self.__class__.visible_activation_module()
        self._hidden_activation = self.__class__.hidden_activation_module()

        self.weights = Parameter(initial_weights or
                                 random_uniform_tensor(self._nr_visible, self._nr_hiddens))

        self.visible_biases = Parameter(initial_visible_biases or
                                        random_uniform_tensor(self._nr_visible))

        self.hidden_biases = Parameter(initial_hidden_biases or
                                       random_uniform_tensor(self._nr_hiddens))

    def _net_visible(self, hidden):
        """ Return the net visible stimulus given a hidden sample vector
        """
        return self.weights.data.matmul(hidden) + self.visible_biases.data

    def _net_hidden(self, visible):
        """ Return the net hidden stimulus given a visible pattern vector
        """
        return visible.matmul(self.weights.data) + self.hidden_biases.data

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

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self._nr_visible) + ' -> ' \
               + str(self._nr_hiddens) + ')'

    forward = sample_h_given_v

    def reconstruct_error_metric(self, metric):
        return lambda x: metric(self.reconstruct(x), x)
