# See LICENSE file for copyright and license details.
import time
from datetime import timedelta

import torch

class Trainer(object):
    def __init__(self, learning_rule):
        self._learning_rule = learning_rule
        self._epoch = 0
        self._hooks = []

    @property
    def epoch(self):
        return self._epoch

    @staticmethod
    def check_batch(batch):
        if not isinstance(batch, torch.Tensor):
            raise Exception("Batchs must be torch.Tensor's")
        if len(batch.shape) <= 1:
            raise Exception("Batch shape must have at least dimension two")

    def _notify(self, msg):
        for hook in self._hooks:
            hook.notify(msg)

    def train(self, batchs, epochs=1):
        if len(batchs) == 0:
            return

        for batch in batchs:
            self.check_batch(batch)

        self._notify('pre_training')

        for _ in range(epochs):
            self._notify('pre_epoch')
            for batch in batchs:
                self._learning_rule.step(batch)
            self._notify('post_epoch')
            self._epoch += 1

        self._notify('post_training')


    def attach(self, hook):
        self._hooks.append(hook)

class TrainingHook:
    """ Base class for all trainer hooks """
    def __init__(self, trainer):
        self._trainer = trainer
        trainer.attach(self)

    def pre_training(self):
        """ Hook invoked before start training """
        pass

    def pre_epoch(self):
        """ Hook invoked before start a training epoch """
        pass

    def post_epoch(self):
        """ Hook invoked after start a training epoch """
        pass

    def post_training(self):
        """ Hook invoked after training finished """
        pass

    def notify(self, msg):
        return getattr(self, msg)()

    @property
    def trainer(self):
        return self._trainer
