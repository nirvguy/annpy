import abc
import torch

class LearningRule(object):
    """ Abstract class for learning rules for train models
    """

    def __init__(self, model):
        """ Constructor

          Args:
            model (nn.Module): Model to be updated
        """
        self._model = model

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def step(self, batch):
        """ Perform next step of the rule """
        pass
