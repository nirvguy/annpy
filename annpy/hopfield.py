# See LICENSE file for copyright and license details.
import torch
from torch.nn import Module, Parameter
from .base import LearningRule

class HebbsRule(LearningRule):
    """ Hebbs Learning Rule for Hopfield models """

    def parameters_for(self, pattern):
        # Returns pattern x pattern^T, pattern
        return torch.ger(pattern, pattern), pattern

    def step(self, patterns):
        """ Fit in one epoch the network parameters to remember patterns

          Args:
            patterns (iterable of torch.Tensor): Patterns
        """
        if len(patterns) == 0:
            return

        for pattern in patterns:
            pattern_weights, pattern_biases = self.parameters_for(pattern)

            self.model.weights.data.add_(torch.Tensor(pattern_weights))
            self.model.biases.data.add_(torch.Tensor(pattern_biases))

        self.model.weights.data.div_(len(patterns))
        self.model.biases.data.div_(len(patterns))

class Hopfield(Module):
    """ Hopfield neural network model with synchronous update of states """

    activation = torch.sign

    def __init__(self, inputs, weights=None, biases=None, dtype=torch.FloatTensor):
        """ Construct the Hopfield networks

          Args:
            inputs (int):           Number of inputs units
            weights (torch.Tensor): Tensor with initial weights with shape `inputs` x `inputs`
            biases (torch.Tensor):  Tensor with inital biases with shape `ìnputs`
            dtype:                  Tensor type. Examples: torch.FloatTensor, torch.cuda.FloatTensor
        """
        super(Hopfield, self).__init__()
        self._input_size = inputs
        self.weights = Parameter(weights or
                                 torch.zeros(self._input_size, self._input_size).type(dtype))

        self.biases = Parameter(biases or
                                torch.zeros(self._input_size).type(dtype))

    def net_output(self, input):
        """ Result of update input before activation

          Args:
            input (torch.Tensor): State input tensor (in {-1, 1})
        """
        return self.weights.data.matmul(input) + self.biases.data

    def _output(self, input):
        """ Synchrounus update of input

          Args:
            input (torch.Tensor): State input tensor (in {-1, 1})
        """
        return self.activation(self.net_output(input))

    def reconstruct(self, input, steps=1):
        """ Retrive pattern from the network

          Args:
            input (torch.Tensor): Input pattern (in {-1, 1})
            steps = Number of updates iterations.

          Returns:
            Retrived state pattern after evolving `input` `steps` iterations
        """
        for _ in range(steps):
            output = self._output(input)
            if (output == input).all():
                break
            input = output
        return input

    forward = reconstruct
